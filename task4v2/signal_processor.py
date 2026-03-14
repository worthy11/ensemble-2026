import cv2
import numpy as np

class SignalPostProcessor:
    """
    Klasa dokonująca heurystycznego post-processingu maski sygnału uzyskanej z modelu Leader.
    Odpowiednia do filtracji szumów generacji, podziału na ROI i interpolacji uciętch ciągów.
    """
    def __init__(self, min_area=30):
        self.min_area = min_area
        
    def filter_noise(self, mask):
        """
        Usuwa drobne punkty szumu z maski używając analizy spójnych komponentów (plam).
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                filtered_mask[labels == i] = 255
        return filtered_mask

    def find_rois(self, mask, num_columns=4):
        """
        Dzieli gotowy, sprostowany obraz maski na układ kolumn odpowiadających 
        poszczególnym regionom odprowadzeń (Region Of Interest np. I, aVR, V1, V4).
        Zwraca listę tupli z zakresem pikseli X (x_start, x_end).
        """
        height, width = mask.shape
        col_width = width // num_columns
        rois = []
        for i in range(num_columns):
            x_start = i * col_width
            x_end = (i + 1) * col_width if i < num_columns - 1 else width
            rois.append((x_start, x_end))
        return rois

    def _find_peaks(self, array, num_peaks=3, min_distance=150):
        """
        Pomocnicza funkcja heurystycznie szukająca konkretnej ilości głównych pików.
        Pozbywa się szumu poprzez ucinanie dystansów dookoła znalezionych wierzchołków.
        """
        peaks = []
        arr_copy = array.copy()
        for _ in range(num_peaks):
            peak = int(np.argmax(arr_copy))
            if arr_copy[peak] == 0: 
                break
            peaks.append(peak)
            
            # Wygazowanie bliskiego otoczenia by nie znalezc zaraz obok w tej samej fali R
            start = max(0, peak - min_distance)
            end = min(len(arr_copy), peak + min_distance)
            arr_copy[start:end] = 0
            
        return sorted(peaks)

    def find_baselines(self, mask, rois, max_leads_per_col=3):
        """
        Wyznacza linie bazowe (izoelektryczne) dla każdego ROI w oparciu o poziomą gęstość pikseli 
        (Histogram zagęszczenia w pionie sumy odprowadzeń - odprowadzenia ciągną się po stałym 'Y').
        """
        baselines = []
        for (x_start, x_end) in rois:
            roi_mask = mask[:, x_start:x_end]
            
            # Suma pikseli w rzędach daje profil poziomy gęstości rysunku. Oś izoelektryczna ma zawsze najwięcej pikseli (jest bazą i najwięcej tam czasu spędza krzywa)
            row_sums = np.sum(roi_mask, axis=1)
            
            # Wygładzenie histogramu by zagęścić i zniwelować skoki szumu rzedów
            smoothed = np.convolve(row_sums, np.ones(30)/30, mode='same')
            
            # Poszukujemy dominujących wierszy (np. 3 odprowadzenia rzędami w 1 kolumnie ROI)
            peaks = self._find_peaks(smoothed, num_peaks=max_leads_per_col)
            baselines.append(peaks)
            
        return baselines

    def reconstruct_gaps(self, mask):
        """
        Wyszukuje końcówki (endpoints) poprzerywanych przez szum/brud/predykcję linii. 
        Ponieważ ślad EKG biegnie w stałym rytmie przez oś X, pionowe cienkie piki są najbardziej narażone.
        Zastosujemy operację Zamknięcia Morfologicznego w wertykali by zasklepić ubytki ostre.
        """
        # Element pionowy - zasklepi pionowe ucięcia (skomplikowanie wysokie i cienkie krzywe R)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 25))
        reconstructed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
        
        # Element poziomy - zasklepi poziome ucięcia bazeline'u gdzie siatka mogła skrzyżować się lub nakryć mocny szum
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, kernel_h)
        
        return reconstructed

    def process(self, leader_mask):
        """
        Główna rura dla post-processingu heurystycznego maski.
        Zwraca pełen tuple (Naprawiony i zdeszumowany obraz 2D, Współrzędne Podziału kolumn ROI, Y-Linie Bazowe)
        """
        # Upewnienie się że to matryca binary
        mask = (leader_mask > 127).astype(np.uint8) * 255
        
        # 1. Filtrowanie szumów
        filtered = self.filter_noise(mask)
        
        # 2. Reperacja przerw gęstego i nagłego sygnału EKG (Reconstruct gaps interpolation via morphologies)
        reconstructed = self.reconstruct_gaps(filtered)
        
        # 3. Dodatkowo znajdujemy baseliny i roi by móc w finalnym kroku transformować na formę czasu V/mV
        rois = self.find_rois(reconstructed)
        baselines = self.find_baselines(reconstructed, rois)
        
        return reconstructed, rois, baselines
