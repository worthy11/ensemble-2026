import cv2
import numpy as np

class Gridder:
    """
    Gridder przetwarza binarną maskę punktów z modelu Dotter i buduje
    matematyczną matrycę siatki, interpolując brakujące punkty.
    """
    def __init__(self, expected_spacing_x=30, expected_spacing_y=30):
        # Oczekiwane odstępy pomiędzy punktami gridu w pikselach (heurystyka)
        self.expected_spacing_x = expected_spacing_x
        self.expected_spacing_y = expected_spacing_y

    def find_points(self, mask):
        """
        Znajduje środki ciężkości plam z maski binarnej.
        """
        # Możliwe, że maska ma wartości 0 i 255. Ensure it is uint8.
        mask = mask.astype(np.uint8)
        
        # Ekstrakcja spójnych komponentów masek (plam) do zrzucenia do listy środków
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Pomiń etykietę tła (index 0)
        points = []
        for i in range(1, num_labels):
            # Możemy odfiltrować za małe plamy szumu opcjonalnie
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 1: # Filtrowanie ultra małego szumu (np 1 px)
                points.append(centroids[i])
                
        return np.array(points)

    def build_grid_matrix(self, points, img_shape):
        """
        Buduje siatkę punktów (wiersze x kolumny) interpolując brakujące i nadając im układ współrzędnych obrazu.
        Jest to uproszczona implementacja polegająca na sortowaniu po osiach Y i X z heurystyką odległościową.
        Złożone zdjęcia wymagają bardziej zaawansowanych algorytmów np. RAN-SAC lub grafowych minimalnych ścieżek.
        """
        if len(points) == 0:
            return np.array([[]])

        # 1. Sortowanie punktów Y (góra -> dół)
        points = points[np.argsort(points[:, 1])]
        
        rows = []
        current_row = [points[0]]
        
        # Grupowanie punktów w wierszach opierając się na różnicy współrzędnej Y
        for pt in points[1:]:
            last_pt = current_row[-1]
            if abs(pt[1] - last_pt[1]) < self.expected_spacing_y * 0.5:
                # Należy jeszcze do tego samego wiersza (zgadza się z tolerancją obrotu zdjęcia)
                current_row.append(pt)
            else:
                # Następny wiersz wykryto
                # Posortuj względem X
                current_row = sorted(current_row, key=lambda p: p[0])
                rows.append(current_row)
                current_row = [pt]
                
        # Ostatni wiersz do dodania
        current_row = sorted(current_row, key=lambda p: p[0])
        rows.append(current_row)

        # 2. Heurystyczna interpolacja brakujących punktów w wierszach (uzupełnianie kolumn)
        # Znajdujemy medianę z ilości punktów w każdym wierszu lub ustalamy twardy limit
        max_cols = max([len(row) for row in rows])
        
        grid_matrix = []
        for row in rows:
            interpolated_row = []
            if len(row) > 0:
                interpolated_row.append(row[0])
                
            for i in range(1, len(row)):
                prev_pt = row[i-1]
                curr_pt = row[i]
                
                dist_x = curr_pt[0] - prev_pt[0]
                
                # Jeśli dystans w X jest nienaturalnie duży (duży brak między punktami X) interpoluj liniowo
                missing_points_n = int(round(dist_x / self.expected_spacing_x)) - 1
                
                for step in range(1, missing_points_n + 1):
                    # Interpolacja liniowa
                    fraction = step / (missing_points_n + 1)
                    interp_x = prev_pt[0] + (curr_pt[0] - prev_pt[0]) * fraction
                    interp_y = prev_pt[1] + (curr_pt[1] - prev_pt[1]) * fraction
                    interpolated_row.append(np.array([interp_x, interp_y]))
                    
                interpolated_row.append(curr_pt)
                
            # Wyrównanie wierszy (padding / ucięcie) dla równej matrycy (bardzo bazowo)
            # Złożniejsze algorytmy przypisują indeks kolumny na bazie uśrednionego offsetu pierwszego X.
            while len(interpolated_row) < max_cols:
                 interpolated_row.append(interpolated_row[-1] + np.array([self.expected_spacing_x, 0])) # Wypychanie prawidlowe koncowek
            
            # Zapinamy rzędy (ucinamy rzędy gdyby overshotnął)
            grid_matrix.append(np.array(interpolated_row[:max_cols]))

        # Zwracamy N x M x 2 macierz współrzędnych gridu
        return np.array(grid_matrix)

    def process(self, mask):
        points = self.find_points(mask)
        grid_matrix = self.build_grid_matrix(points, mask.shape)
        return grid_matrix


class Undistortion:
    """
    Undistortion prostuje obraz na bazie wygenerowanej matrycy punktów (Gridder).
    Wykorzystuje transformaty wizualne 4-punktowe niezależnie dla każdej 'kratki' aby uniknąć dystorsji.
    """
    def __init__(self, target_cell_width=30, target_cell_height=30):
        self.target_cell_width = target_cell_width
        self.target_cell_height = target_cell_height

    def process(self, image, grid_matrix):
        """
        Zwraca scalony wyprostowany obraz EKG.
        image: oryginalny, zaszumiony / zakrzywiony obraz (np. w BGR z openCV)
        grid_matrix: numPy matryca punktów [Rows, Cols, 2] od Griddera z punktami w osiach (X, Y)
        """
        if grid_matrix.ndim != 3 or grid_matrix.shape[2] != 2:
            raise ValueError(f"Nieprawidłowy kształt matrycy (Oczekiwano NxMx2): Otrzymano {grid_matrix.shape}")

        rows, cols, _ = grid_matrix.shape
        
        if rows < 2 or cols < 2:
            # Nie ma wystarczającej matrycy do sprostowania chociaćby 1 kratki.
            print("[WARN] Undistortion: Zbyt mała matryca krat by przeprowadzić prostowanie.")
            return image
        
        # Wyjściowa pustą plansza odpowiadająca równej perspektywie
        out_height = (rows - 1) * self.target_cell_height
        out_width = (cols - 1) * self.target_cell_width
        unwarped_image = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        # Każde 4 punkty idealnego układu dla pojedynczej kratki (gdzie lądują)
        dst_pts = np.float32([
            [0, 0],
            [self.target_cell_width, 0],
            [self.target_cell_width, self.target_cell_height],
            [0, self.target_cell_height]
        ])

        # Iteruj kwadrat po kwadracie z grid matrix i wyciągaj po 4 punkty wejścia
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Współrzędne źródłowe czworokąta
                pt_tl = grid_matrix[i, j]         # Top left
                pt_tr = grid_matrix[i, j+1]       # Top right
                pt_br = grid_matrix[i+1, j+1]     # Bottom right
                pt_bl = grid_matrix[i+1, j]       # Bottom left
                
                src_pts = np.float32([pt_tl, pt_tr, pt_br, pt_bl])
                
                # Ustal perspektywe by spłaszczyć i wyśrodkować na 'raty' dla pojedynczej kratki na płasko
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Transformuj całego EKG dla docelowej kratki, utnie nam nadmiar i dopasuje odchyły rzędu kilku pikseli
                warped_cell = cv2.warpPerspective(image, M, (self.target_cell_width, self.target_cell_height))
                
                # Kopiuj uzyskaną idealną wyprostowaną spasteryzowaną łatę do globalnego wyjściowego obrazu odrestaurowanej matrycy
                y_start, y_end = i * self.target_cell_height, (i + 1) * self.target_cell_height
                x_start, x_end = j * self.target_cell_width,  (j + 1) * self.target_cell_width
                
                unwarped_image[y_start:y_end, x_start:x_end] = warped_cell
                
        return unwarped_image

if __name__ == "__main__":
    # Szybki test by zweryfikować syntaks (nie wywoła wizualnie)
    mock_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mock_mask, (50, 50), 2, 255, -1)
    cv2.circle(mock_mask, (100, 50), 2, 255, -1)
    cv2.circle(mock_mask, (50, 100), 2, 255, -1)
    cv2.circle(mock_mask, (100, 100), 2, 255, -1)
    
    mock_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    gridder = Gridder()
    grid_mat = gridder.process(mock_mask)
    print("Grid Matrix Shape:", grid_mat.shape)
    
    undistorter = Undistortion()
    final_img = undistorter.process(mock_image, grid_mat)
    print("Final Unwarped Image Shape:", final_img.shape)
