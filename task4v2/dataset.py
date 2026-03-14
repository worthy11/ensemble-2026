import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    """
    Dedykowany dataset PyTorch do ładowania obrazów EKG z folderu wejściowego
    zgodnego z optymalnym potokiem ładowania przez bibliotekę OpenCV.
    """
    def __init__(self, image_dir, is_train=False):
        """
        image_dir: Ścieżka do folderu ze zdjęciami testowymi.
        """
        self.image_dir = image_dir
        self.is_train = is_train
        # Pobieranie listy tylko plików będących standardowymi formatami obrazu i niebędących ukrytymi
        self.image_files = [f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Wczytujemy obraz EKG jako macierz pikseli (za pomocą OpenCV)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Nie można załadować obrazu pod ścieżką: {img_path}")
        
        # Zmiana układu kolorów domyślnego dla OpenCV (BGR -> RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Opcjonalnie formatowanie/zmiana wielkości tutaj (w celu skalowania wejścia do wspólnego wymiaru)
        # Przykład, jeżeli sieć wymaga powtarzalnych stałych wejść, można usunąć komentarz:
        # image = cv2.resize(image, (512, 512))
        
        # Szybka macierzowa normalizacja danych [0-255] -> zakrs (0.0-1.0)
        image = image.astype(np.float32) / 255.0
        
        # Transpozycja na format akceptowalny w PyTorch tzn. (Kanały, Wysokość, Szerokość) = (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Konwersja z ndarray OpenCV na Tensor w PyTorch
        image_tensor = torch.tensor(image)
        
        if self.is_train:
            # Ponieważ klasyczny dataset nie ma etykiet dla SIATKI PIKSELOWEJ the Dottera, 
            # na ten moment zwracamy wygenerowany syntetyczny / maskujący czarny ekran zastępczy (lub z-algorytmizowany grid).
            _, h, w = image_tensor.shape
            dummy_mask = torch.zeros((1, h, w), dtype=torch.float32)
            return image_tensor, dummy_mask, self.image_files[idx]
        
        return image_tensor, self.image_files[idx]
