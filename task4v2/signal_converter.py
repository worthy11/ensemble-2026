import numpy as np
from scipy.interpolate import interp1d

class SignalConverter:
    """
    Klasa dokonująca cyfrowej konwersji (Digitalizacji 1D) ze spłaszczonej maski sybilnego sygnału
    na ustrukturyzowane wielkości fizyczne osi czasu i napięcia wedle medycznych restrykcji EKG.
    
    Zakłada parametry gridu:
    - 25 mm / sekundę
    - 10 mm / mV
    - Target frequency: 500 Hz
    """
    def __init__(self, pixels_per_mm=30, target_hz=500, paper_speed_mm_s=25, gain_mm_mv=10):
        self.pixels_per_mm = pixels_per_mm
        self.target_hz = target_hz
        self.paper_speed = paper_speed_mm_s
        self.gain = gain_mm_mv
        
        # Obliczenia bazowe
        # Przelicznik: Ile pikseli wynosi 1 sekunda sygnału w osi X
        self.pixels_per_sec = self.pixels_per_mm * self.paper_speed
        
        # Przelicznik: Ile pikseli w osi Y wynosi 1 mV napięcia
        self.pixels_per_mv = self.pixels_per_mm * self.gain

        # Standardowy europejski 12-odprowadzeniowy format na papierze 4 kolumnowym, 3 rzędowym
        # Kolumna1 = I, II, III. Kolumna2 = aVR, aVL, aVF. Kolumna3 = V1, V2, V3. Kolumna4 = V4, V5, V6
        self.STANDARD_LEAD_LAYOUT = [
            ['I', 'II', 'III'],
            ['aVR', 'aVL', 'aVF'],
            ['V1', 'V2', 'V3'],
            ['V4', 'V5', 'V6']
        ]

    def _extract_1d_signal(self, sub_mask, baseline_y):
        """
        Zmienia dwuwymiarowy ślad (plamy maski) w matematyczną funkcję 1D x -> y
        poprzez uśrednienie pikseli maski dla każdego kroku współrzędnej X obrazu EKG.
        Zwraca listę odchyleń od baseline_y w pikselach.
        """
        _, width = sub_mask.shape
        signal_pixels_y = []
        
        for x in range(width):
            # Znajdź wszystkie włączone piksle (255) w kolumnie X
            col = sub_mask[:, x]
            y_indices = np.where(col > 0)[0]
            
            if len(y_indices) > 0:
                # Bierzemy średnią (środek) grubości wydrukowanej ścieżki (albo maski predykcyjnej Lidera)
                avg_y = np.mean(y_indices)
                
                # Zwróć odchylenie. Pamiętaj: W OpenCV w osi Y piksele rosną w dół. 
                # (Sygnał ujemny na EKG jest niżej na papierze tzn. ma większe Y od baseline_y)
                # Zatem: deviation = baseline_y - avg_y
                deviation_from_base = baseline_y - avg_y 
                signal_pixels_y.append(deviation_from_base)
            else:
                # Jeśli Leader zignorował albo brud usunął ciągłość, musimy założyć nan do interpolacji,
                # albo wrzucić zera (płaski baseline) jeśli dziura jest krótka.
                signal_pixels_y.append(np.nan)

        return np.array(signal_pixels_y)

    def _convert_to_physical(self, signal_deviation_px):
        """
        Zamienia odchylenia w pikselach na milivolty i konwertuje X z próbek pikseli na 500 Hz.
        """
        # Interpolacja nielicznych braków wyciągniętych przez post process na bazie NaN
        nans, x = np.isnan(signal_deviation_px), lambda z: z.nonzero()[0]
        if np.any(nans):
            signal_deviation_px[nans] = np.interp(x(nans), x(~nans), signal_deviation_px[~nans])
            
        # 1. Konwersja osi Y (Amplituda: piksele -> mV)
        # Napięcie (mV) = odchylenie_w_pikselach / piksele_na_1_mV
        signal_mv = signal_deviation_px / self.pixels_per_mv
        
        # 2. Konwersja osi X (Czas: częstotliwość rozdzielczości pikseli -> częstotliwość równa 500Hz)
        # Długość sygnału wynosi len(signal_mv) w pikselach.
        # Skoro wiemy, że X pikseli to 1 sekunda (self.pixels_per_sec), czas trwania t = len / pixels_per_sec.
        total_time_seconds = len(signal_mv) / self.pixels_per_sec
        
        # Nową pożądaną ilością powszechnie akceptowalną sztywną ilością punktów jest target_hz * całkowity_czas
        num_target_samples = int(total_time_seconds * self.target_hz)
        
        # Oś czasu wejściowa (indeksy równo oddalone w pikselach)
        old_x = np.linspace(0, total_time_seconds, len(signal_mv))
        # Oś czasu re-samplingowa (równy raster próbkowania dla 500 Hz)
        new_x = np.linspace(0, total_time_seconds, num_target_samples)
        
        # Interpolacja splajnowa (lub liniowa) z wykorzystaniem scipy do równego 500Hz
        # Uzywamy 'linear' domyślnie, można też użyć krawędzi 'cubic' przy wygładzaniu większych dziur
        interpolator = interp1d(old_x, signal_mv, kind='linear', bounds_error=False, fill_value="extrapolate")
        resampled_signal_mv = interpolator(new_x)

        # Rzutowanie sygnału na typ float16 dla znaczącej redukcji objętości danych .npz (szczególnie przy wielu obrazach)
        return resampled_signal_mv.astype(np.float16)

    def process(self, final_mask, rois, baselines):
        """
        Główna rura dla całego obrazu. Wciąga maskę z rois oraz wyznaczonymi baseline'ami.
        Dokonuje weryfikacji i konwersji fizycznej zwracając ostateczny słownik:
        dict: {'I': [...], 'II': [...], 'V1': [...], ...} z tablicami numpy dla danego odprowadzenia napięć
        """
        output_signals = {}
        
        # Zakładamy poprawność układu 12-lead (4 columns x 3 rows) dla uproszczenia
        for col_idx, ((x_start, x_end), baseline_list) in enumerate(zip(rois, baselines)):
            
            # W danym pionowym Region Of Interest wyciągniętym z maski
            col_mask = final_mask[:, x_start:x_end]
            height = col_mask.shape[0]
            
            # Tworzymy podział wysokości kolumny (na rzędy) bazując równomiernie na baseline'ach
            # Można również dzielić sztywno na wysokość/3 jeżeli są duże wahania osi, ale użyjmy baseline index
            
            # Wyznacz mid-pointy miedzy baselines zeby nie poosiągac innych linii w obszar 1D
            slice_bounds = [0]
            for row_idx in range(len(baseline_list) - 1):
                mid = (baseline_list[row_idx] + baseline_list[row_idx+1]) // 2
                slice_bounds.append(mid)
            slice_bounds.append(height)

            for row_idx in range(len(baseline_list)):
                if col_idx < len(self.STANDARD_LEAD_LAYOUT) and row_idx < len(self.STANDARD_LEAD_LAYOUT[col_idx]):
                    lead_name = self.STANDARD_LEAD_LAYOUT[col_idx][row_idx]
                else:
                    lead_name = f"UKNN_C{col_idx}R{row_idx}" # Rezerwowy Unknown handling
                    
                # Wyciągnij precyzyjny kwadracik zawierajacy 1 ślad z uciętym tłem do limitów kolizyjnćh
                y_s = slice_bounds[row_idx]
                y_e = slice_bounds[row_idx+1]
                
                # Zabezpiecznie przez ewentualne baseliny przylegle w prozie
                roi_trace_mask = col_mask[y_s:y_e, :]
                local_base_y = baseline_list[row_idx] - y_s # Znormalizowanie baseline Y odgórnie ucinki do układu krawędzi łaty obrazowej
                
                # Digitalizuj: Piksele uciętej binarnej szyny -> 1D lista odchyleń od osi z wyrównaniem offsetu X dla pustych
                trace_1d_px = self._extract_1d_signal(roi_trace_mask, local_base_y)
                
                # Konwertuj matematycznie: Amplitudy pikseli na MV, Oś X próbki obiektywu na Sekundy@500Hz
                physical_trace_mv = self._convert_to_physical(trace_1d_px)
                
                output_signals[lead_name] = physical_trace_mv
        
        return output_signals
