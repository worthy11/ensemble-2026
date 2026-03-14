import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ECGDataset  # Na razie korzystamy ze starego, zaraz go zaktualizujemy do zwracania Masek Ground Truth
from dotter_model import DotterModel

# Podstawowa funkcja straty do segmentacji binarnej
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        # BCE z Logits jest stabilniejsze numerycznie (nie wymaga Sigmoida na wyjściu z sieci)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # inputs: surowe wyjścia (logits) z modelu
        bce_loss = self.bce(inputs, targets)
        
        # Do Dice loss aplikujemy Sigmoid, żeby sprowadzić logits do zakresu [0, 1]
        inputs = torch.sigmoid(inputs)
        
        # Spłaszczenie tensorów do wektorów
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return bce_loss + dice_loss

def train_model(args):
    """
    Główna pętla ucząca w podziale na **Batche** oraz **Epoki**.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN] Odtwarzanie układu: {device.type.upper()}")
    
    # Krok 1: Inicjalizacja Modelu
    # Trenujemy na razie "The Dotter" (sieć od siatki). Model ma 3 kanały wejściowe i 1 wyjściowy.
    model = DotterModel(in_channels=3, out_channels=1).to(device)
    
    # Opcjonalnie: załadowanie poprzednich wag do dotrenowywania (transfer learning)
    if args.resume and os.path.exists(args.weights_out):
        model.load_state_dict(torch.load(args.weights_out, map_location=device))
        print(f"[TRAIN] Wczytano wagi początkowe z {args.weights_out}")

    # Krok 2: Dataset i DataLoaders z PODZIAŁEM NA BATCHE
    train_dataset = ECGDataset(args.input_dir, is_train=True) # is_train doda logikę GT w dataset.py
    
    # DataLoader tnie nam te 3000 zdjęć na "Paczki" (Batches). 
    # batch_size=8 oznacza, że karta graficzna przeliczy najpierw 8 zdjęć naraz na 1 krok aktualizacji.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Bardzo ważne - losujemy kolejność w epokach żeby zapobiec pamięciowemu przeuczeniu
        num_workers=4, # Multithreading wczytywania CPU->RAM
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"[TRAIN] Wczytano dataset: {len(train_dataset)} obrazów. Rozmiar Batcha: {args.batch_size}.")
    print(f"[TRAIN] Liczba wymuszonych aktualizacji gradientu (kroków) na Epokę: {len(train_loader)}")

    # Krok 3: Optymalizator i Funkcja Straty
    criterion = DiceBCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW z małą karą za ogromne wagi (L2)

    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    # Krok 4: PĘTLA EPOK
    print("\n--- ROZPOCZĘCIE TRENINGU ---")
    for epoch in range(1, args.epochs + 1):
        model.train() # Uruchomienie trybu treningowego (uruchamia np. Batch Normalization i Dropout)
        epoch_loss = 0.0
        
        # Pasek postępu dla przetwarzania wsadowego (Batch processing) w pojedynczej Epoce
        with tqdm(total=len(train_loader), desc=f"Epoka {epoch}/{args.epochs}", unit="batch") as pbar:
            for images, masks, filenames in train_loader:
                # Wyrzucenie batcha paczki wejściowej na RAM karty CUDA
                images = images.to(device)
                masks = masks.to(device)
                
                # Zeroing gradientów z poprzedniego kroku (konieczne w PyTorch)
                optimizer.zero_grad()
                
                # 1. Forward Pass (Predykcja paczki zdjęć przez Model)
                preds = model(images)
                
                # 2. Obliczanie błędu (Loss) dla całej paczki naraz
                loss = criterion(preds, masks)
                
                # 3. Backward Pass (Kombinatoryka pochodych - Wsteczna propagacja błędów)
                loss.backward()
                
                # 4. Optymalizacja (Fizyczne przesunięcie wag modelu w stronę poprawy)
                optimizer.step()
                
                # Statystyki
                epoch_loss += loss.item()
                pbar.set_postfix(**{"Loss": loss.item()})
                pbar.update(1)
                
        # Podsumowanie uśrednionej Epoki
        avg_loss = epoch_loss / len(train_loader)
        print(f"-> Zakończono Epokę {epoch}. Średni błąd iteracji (Loss): {avg_loss:.4f}")
        
        # Opcjonalnie zapis co N epok
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.weights_out)
            print(f"-> Zapisano check-point punktowy modelu do {args.weights_out}")

    print("\n[TRAIN] Nauka The Dotter zakończona sukcesem! Skrypt inference.py może konsumować wyniki.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pętla Treningowa (Epochs/Batches) dla modelu")
    parser.add_argument("--input_dir", type=str, default="data/train", help="Folder ze zdjęciami i ground-truth")
    parser.add_argument("--batch_size", type=int, default=4, help="Ilość pikseli/zdjęć ładownych do VRAM karty graficznej naraz")
    parser.add_argument("--epochs", type=int, default=30, help="Ilość pełnych powtórzeń nauki na całym zbiorze danych")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate - Rozmiar pojedynczego kroku optymalizatora")
    parser.add_argument("--weights_out", type=str, default="weights/dotter_weights.pth", help="Gdzie zapisać nauczony mózg")
    parser.add_argument("--resume", action="store_true", help="Podnieś wagi z dotter_weights.pth żeby kontynuować przerwany trening")
    
    args = parser.parse_args()
    train_model(args)
