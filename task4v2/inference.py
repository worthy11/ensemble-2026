import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset import ECGDataset
from grid_processor import Gridder, Undistortion
import torch.nn.functional as F
from leader_model import LeaderModel
from signal_processor import SignalPostProcessor
from signal_converter import SignalConverter

# ==============================================================================
import argparse

# Opcjonalna stała: Ścieżka do wytrenowanych wag (jeśli dostępne)
# MODEL_WEIGHTS_PATH = "weights/dotter_weights.pth" 

def main():
    parser = argparse.ArgumentParser(description="Uruchamia potok dygitalizacji EKG")
    parser.add_argument("--input", type=str, default="data/input_ekg_images", help="Folder ze zdjęciami wejściowymi (np. 'test/' lub 'train/')")
    parser.add_argument("--output", type=str, default="data/out", help="Folder docelowy dla wygenerowanego pliku submission.npz")
    args = parser.parse_args()
    
    INPUT_IMAGE_DIR = args.input
    OUTPUT_NPZ_DIR = args.output
    
    # Tymczasowe foldery robocze (maski i sygnały) można zapisać jako podfoldery w output
    OUTPUT_MASK_DIR = os.path.join(args.output, "output_bin_masks")
    OUTPUT_UNWARPED_DIR = os.path.join(args.output, "output_unwarped")
    OUTPUT_SIGNAL_DIR = os.path.join(args.output, "output_signals")
    # 1. Optymalizacja PyTorch - wykorzystywanie platformy Cuda od Nvidia dla akceleracji (lub fallback do CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Backend obliczeniowy uruchomiony w profilu: {device.type.upper()}")

    # 2. Inicjalizacja optymalnego modelu (architektura u-net z ustrukturyzowaniem opartym o res-blocks)
    model = DotterModel(in_channels=3, out_channels=1)
    leader_model = LeaderModel(in_channels=3, out_channels=1)
    
    # Opcjonalna faza dla użytkownika w przypadku istnienia wag: (odkomentuj po wyuczeniu wag)
    # if os.path.exists(MODEL_WEIGHTS_PATH):
    #     model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    #     print("[INFO] Wczytano wagi do modelu.")
    # else:
    #     print("[WARN] Korzystam z losowych wag wejściowych (szkielet z niepołączonymi wagami), uzyskane segmentacje to demonstracja inferencji i pipeline'u.")

    model = model.to(device)
    model.eval() # Zabezpieczenie przed zachowaniem dropoutu/bn w trybie prediction.

    leader_model = leader_model.to(device)
    leader_model.eval()

    # 3. Definiowanie folderów przez os.makedirs dla bezpieczeństwa
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_UNWARPED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SIGNAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
    
    dataset = ECGDataset(INPUT_IMAGE_DIR)
    if len(dataset) == 0:
        print(f"[WARN] Aktualnie pod folderem {INPUT_IMAGE_DIR} nie znajduje się żadne zdjęcie.")
        print("Modyfikuj plik wejściowy 'INPUT_IMAGE_DIR' podając odpowiednią ścieżkę EKG i spróbuj ponownie.")
        return

    # Multithreading dzięki dataloader pod warunkiem ze sprzet i cuda umozliwia (num_workers, pin_memory)
    # pin_memory = True maksymalizuje przeslanie tensora host -> GPU co stanowi znaczne wsparcie Cuda
    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)

    gridder = Gridder()
    undistorter = Undistortion()
    signal_processor = SignalPostProcessor()
    
    # Inicjalizacja konwertera, na ogół z sirodowiska medycznego standardowe jest 30 pixeli/mm po operacjach na unwarped
    # Dostosuj 'pixels_per_mm' zależnie od faktycznego docelowego skalowania po Undistortion by zachować skalę fizyczną.
    signal_converter = SignalConverter(pixels_per_mm=30, target_hz=500, paper_speed_mm_s=25, gain_mm_mv=10)

    # Słownik globalny (płaski format) na predykcje i serializację testową
    flat_submission_dict = {}

    # 4. Inferencja po danych
    print(f"[INFO] Rozpoczynanie predykcji segmentacyjnej masek na {len(dataset)} plikach.")
    with torch.no_grad():
        for i, (images, filenames) in enumerate(dataloader):
            images = images.to(device)
            filename = filenames[0]
            
            # Wystosowanie estymacji w sieci opartej o res blocks (uzyskamy (Batch, Kanały, X, Y))
            preds = model(images)
            
            # Spłaszczenie struktury tensorialnej do prostej surowej macierzy formatu wymiarowego klasycznego rastra graficznego [X][Y]
            preds_np = preds.squeeze().cpu().numpy()
            
            # Binaryzacja (z racji, że return oparty o Sigmoid wyrzuca liczby bliskie 1 lub 0) na standardach decyzyjnych progu .5 - Zwracanie "czarno bialej maski"
            binary_mask = (preds_np > 0.5).astype(np.uint8) * 255
            
            # Zapis maski
            output_file_name_mask = f"mask_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, output_file_name_mask), binary_mask)
            
            # URUCHOMIENIE PIPELINE'U GRIDDERA I UNDISTORTION
            print(f" -> Analiza siatki Gridder dla {filename}...")
            # Ponieważ dataset zwraca tensor RGB HWC normalnie, musimy doczytać oryginał żeby użyć Undistorter na kolorowym BGR z openCV bez ucinania
            original_image = cv2.imread(os.path.join(INPUT_IMAGE_DIR, filename))

            # 1. Pobranie matrycy krzyżowań z maski The Dotter'a
            grid_matrix = gridder.process(binary_mask)
            
            # 2. Wyprostowanie EKG po siatce i złożki dla krat
            print(f" -> Wykonywanie 4-punktowej transformacji OpenCV na matrycy dla {filename}...")
            unwarped_ecg = undistorter.process(original_image, grid_matrix)
            
            # Zapis gotowego wyprostowanego EKG
            output_file_name_unwarped = f"unwarped_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_UNWARPED_DIR, output_file_name_unwarped), unwarped_ecg)

            # --- LEADER MODEL & SYGNAŁ POST_PROCESSING ---
            print(f" -> Predykcja precyzyjnych śladów sygnału EKG (Leader) dla {filename}...")
            # Konwersja OpenCV BGR numpy do formy zjadliwej dla PyTorch Tensor SMP Resnet (Batch, Channels, H, W)
            unwarped_rgb = cv2.cvtColor(unwarped_ecg, cv2.COLOR_BGR2RGB)
            unwarped_np = (unwarped_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            unwarped_tensor = torch.tensor(unwarped_np).unsqueeze(0).to(device)
            
            # Sieci U-Net ze względu na pooling/upsampling w SMP i generalnie dekoderach wymagaj aby wymiary 
            # dzieliły się przez 32 (stąd padujemy na dole lub w boku jeżeli po wyprostowaniu otrzymamy głupią lub utratną liczbę)
            h, w = unwarped_tensor.shape[2], unwarped_tensor.shape[3]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                unwarped_tensor = F.pad(unwarped_tensor, (0, pad_w, 0, pad_h))

            # Predykcja
            leader_preds = leader_model(unwarped_tensor)
            
            # Odcięcie wepchanego padu i cofnięcie do zbieżnego i zgranego oryginalnego wymiaru uwarped
            if pad_h > 0 or pad_w > 0:
                leader_preds = leader_preds[:, :, :h, :w]
            
            leader_preds_np = leader_preds.squeeze().cpu().numpy()
            
            # Generacja czarno-białej binarnej maski samych sygnałów i izolacja!
            leader_mask = (leader_preds_np > 0.5).astype(np.uint8) * 255
            
            # Heurystyki post processingu odszumianie, odbudowa, baseliny itp
            print(f" -> Heurystyki post-processingu sygnałów i ekstrakcja danych metrycznych...")
            final_signal_mask, rois, baselines = signal_processor.process(leader_mask)
            
            # Zapis ostatecznie gotowej maski signal 
            output_file_name_signals = f"signal_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_SIGNAL_DIR, output_file_name_signals), final_signal_mask)

            # --- DIGITALIZACJA I KONWERSJA FIZYCZNA ---
            print(f" -> Cyfryzacja sygnału 1D i konwersja (piksele -> mV / 500Hz) dla {filename}...")
            
            # Mapowanie wysepek sygnałoych na fizyczne wektory przypisanych odprowadzeń
            physical_leads_dict = signal_converter.process(final_signal_mask, rois, baselines)
            
            # Dodanie wylistowanych odprowadzeń jako kluczy płaskich ({record}_{lead}) 
            record_name = os.path.splitext(filename)[0]
            for lead_name, signal_1d in physical_leads_dict.items():
                flat_key = f"{record_name}_{lead_name}"
                flat_submission_dict[flat_key] = signal_1d

            print(f" -> [SUKCES] Zdigitalizowano ślady dla {record_name}.")

    # Generacja ostatecznego spakowanego słownika pod submission.npz na koniec pracy skryptu
    final_npz_path = os.path.join(OUTPUT_NPZ_DIR, "submission.npz")
    np.savez(final_npz_path, **flat_submission_dict)

    print(f"\n[SUCCESS] Cały ekosystem EKG zakończył obróbkę plików z {INPUT_IMAGE_DIR}!")
    print(f" * WYGENEROWANE MASKI SIATKI (DOTTER): {OUTPUT_MASK_DIR}")
    print(f" * WYPROSTOWANE ZDJĘCIA EKG: {OUTPUT_UNWARPED_DIR}")
    print(f" * HEURYSTYKI ODTWORZONEGO SYGNALU (LEADER): {OUTPUT_SIGNAL_DIR}")
    print(f" * DIGITALIZACJA 1D GŁÓWNY WYNIK: {final_npz_path}")

if __name__ == "__main__":
    main()
