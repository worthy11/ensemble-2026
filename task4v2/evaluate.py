import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import json
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags

from dotter_model import DotterModel
from grid_processor import Gridder, Undistortion
from leader_model import LeaderModel
from signal_processor import SignalPostProcessor
from signal_converter import SignalConverter
from wfdb_loader import load_ground_truth

def pad_tensor_for_unet(tensor):
    """Pads tensor so H and W are divisible by 32 (smp UNet config)."""
    h, w = tensor.shape[2], tensor.shape[3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor, h, w, pad_h, pad_w

def calculate_metrics(y_true, y_pred, fs=500):
    """
    Kalkuluje metryki w oparciu o ostateczną specyfikację turniejową/ewaluacyjną.
    Zwraca słownik punktów cząstkowych i informacji do wyświetlenia.
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return {"shape_pts": 0, "amp_pts": 0, "time_pts": 0, "total": 0, "logs": "Brak danych"}

    # 1. Signal Shape (Max 60 pkt) - Korelacja Pearsona
    if np.var(y_true) > 0 and np.var(y_pred) > 0:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = 0.0
    shape_pts = max(0, corr) * 60.0 
    
    # 2. Amplitude (Max 20 pkt) - SNR
    noise = y_true - y_pred
    noise_power = np.sum(noise ** 2)
    signal_power = np.sum(y_true ** 2)
    
    if noise_power == 0:
        snr_db = 30.0 
    else:
        snr_db = 10 * np.log10(signal_power / noise_power)
        
    capped_snr = max(0, min(snr_db, 20.0))
    amp_pts = capped_snr
    
    # 3. Time Calibration (Max 20 pkt) - Cross-Correlation
    correlation = correlate(y_true, y_pred, mode='full')
    lags = correlation_lags(len(y_true), len(y_pred), mode='full')
    best_lag = lags[np.argmax(correlation)]
    
    lag_ms = (abs(best_lag) / fs) * 1000.0
    time_pts = max(0, 20.0 - (lag_ms / 5.0))
    
    total_pts = shape_pts + amp_pts + time_pts
    return {
        "shape_pts": shape_pts, "pearson": corr,
        "amp_pts": amp_pts, "snr": snr_db,
        "time_pts": time_pts, "lag_ms": lag_ms,
        "total": total_pts
    }

def run_evaluation_pipeline(image_path, json_path, wfdb_record_path):
    """
    Przeprowadza walidację End-to-End potoku wizyjnego względem 
    twardego, zmatematyzowanego serwomatora GT z formy WFDB przydzielanego na 500Hz.
    
    Zgodnie ze specyfikacją:
    .json zawiera współrzędne pikselowe przydzielone TYLKO do czystego, WYPROSTOWANEGO obrazu z kroku Undistortion.
    .dat/.hea (wfdb) to finalny cel w postaci szeregu czasowego napięć 1D w mV @ 500 Hz.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Ładowanie i predykcja modelu Dotter (Wyłuskiwanie siatki pre-processing)
    raw_img = cv2.imread(image_path)
    # Convert BGR -> RGB Tensor format Normalized
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_np = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)
    
    dotter = DotterModel().to(device).eval()
    leader = LeaderModel().to(device).eval()
    
    with torch.no_grad():
        preds_grid = dotter(img_tensor).squeeze().cpu().numpy()
        binary_mask_grid = (preds_grid > 0.5).astype(np.uint8) * 255
        
    # 2. Gridder & Undistortion (Normalizacja / Odkręcanie zniekształceń perspektywy)
    gridder = Gridder()
    undistorter = Undistortion()
    grid_matrix = gridder.process(binary_mask_grid)
    unwarped_ecg = undistorter.process(raw_img, grid_matrix) # TO JEST CEL DLA JSON
    
    # --- [SPECYFIKACJA]: Ograniczenia i logika plików JSON nakładana jest na sprostowany wycinek 'unwarped_ecg' ---
    # Jeżeli model miałby się trenować / optymalizować częściowo na koordydatach z JSON (np. w systemie bbox / RCNN),
    # To właśnie z tego miejsca (na tym unwarped_ecg matrycy np. 1200x2000 px) te labelki pokryją się perfekcyjnie z plamą. 
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            pixel_labels = json.load(f)
            # pseudo-logic do obsługi strat wizjowych opinii 
            # print(f"[JSON] Wczytano {len(pixel_labels)} etykiet współrzędnych do rzutowania na wyprostowany EKG.")
    
    # 3. Model Leader (Wycinanie struktury sygnałowej linii odprowadzeń po z-unwarpowanym wejściu)
    unwarped_rgb = cv2.cvtColor(unwarped_ecg, cv2.COLOR_BGR2RGB)
    unwarped_tensor_np = (unwarped_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    unwarped_tensor_t = torch.tensor(unwarped_tensor_np).unsqueeze(0).to(device)
    
    padded_unwarped, un_h, un_w, ph, pw = pad_tensor_for_unet(unwarped_tensor_t)
    
    with torch.no_grad():
        leader_preds = leader(padded_unwarped)
        if ph > 0 or pw > 0:
            leader_preds = leader_preds[:, :, :un_h, :un_w]
            
        leader_preds_np = leader_preds.squeeze().cpu().numpy()
        leader_mask = (leader_preds_np > 0.5).astype(np.uint8) * 255
        
    # 4. Post-Processing & Konwersja (Heurystyki szumowe i translacja na oś CV-czasu-Napięcia)
    signal_processor = SignalPostProcessor()
    final_signal_mask, rois, baselines = signal_processor.process(leader_mask)
    
    signal_converter = SignalConverter()
    predicted_signals_dict = signal_converter.process(final_signal_mask, rois, baselines)
    
    # 5. Odczyt Prawdziwego referencyjnego źródła Ground Truth (WFDB Loader)
    gt_signals_dict = load_ground_truth(wfdb_record_path)
    
    # 6. Kalkulacja Punktacji na podstawie Specyfikacji (Pearson, SNR, Cross-Correlation)
    print("\n" + "="*60)
    print(f"{'RAPORT PUNKTY EWALUACJI EKG (Max 100 pts)':^60}")
    print("="*60)
    
    total_score_sum = 0
    leads_evaluated = 0
    
    for lead_name in gt_signals_dict.keys():
        if lead_name in predicted_signals_dict:
            y_true = gt_signals_dict[lead_name]
            y_pred = predicted_signals_dict[lead_name]
            
            # Upewnienie się że długość rzutów czasowych @500Hz jest równa
            min_len = min(len(y_true), len(y_pred))
            
            y_true_matched = y_true[:min_len]
            y_pred_matched = y_pred[:min_len]
            
            # Filtrowanie NaN wygenerowanych przez Converter
            valid_mask = ~np.isnan(y_pred_matched)
            
            if not np.any(valid_mask):
                 print(f"{lead_name}: [POMINIĘTO] Brak czytelnych predykcji ciągłych sygnału.")
                 continue
            
            y_true_clean = y_true_matched[valid_mask]
            y_pred_clean = y_pred_matched[valid_mask]
            
            # Ewaluacja algorytmami scipy (Pearson, SNR, Lags)
            metrics = calculate_metrics(y_true_clean, y_pred_clean, fs=500)
            
            leads_evaluated += 1
            total_score_sum += metrics['total']
            
            print(f"Odprowadzenie {lead_name:>3}: {metrics['total']:>5.1f} / 100 pkt")
            print(f"   -> [Shape]: {metrics['shape_pts']:>5.1f}/60 | Pearson = {metrics['pearson']:.3f}")
            print(f"   -> [Ampl ]: {metrics['amp_pts']:>5.1f}/20 | SNR = {metrics['snr']:.1f} dB")
            print(f"   -> [Time ]: {metrics['time_pts']:>5.1f}/20 | Shift = {metrics['lag_ms']:.1f} ms")
            print("-" * 60)
        else:
            print(f"Odprowadzenie {lead_name:>3}: Brak wyprodukowanej osi (Nie odnaleziono ROI Layoutu)")
            
    if leads_evaluated > 0:
        avg_score = total_score_sum / leads_evaluated
        print(f"\n[PODSUMOWANIE] Średni wynik punktowy obrazu: {avg_score:.2f} / 100 pts\n")
    
if __name__ == "__main__":
    # Szybki mock odpalenia (wymaga fizycznych plików WFDB w rejonie)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/input_ekg_images/sample.png")
    parser.add_argument("--json", default="data/labels/sample.json")
    parser.add_argument("--wfdb", default="data/gt/sample") # Odniesie się do .dat / .hea z auto
    args = parser.parse_args()
    
    # Zabezpieczenie na wypadek gdy test fizycznie uzytkownika ruszy w srodowisku bez tych sampli
    if os.path.exists(args.image) and os.path.exists(f"{args.wfdb}.dat"):
        run_evaluation_pipeline(args.image, args.json, args.wfdb)
    else:
        print("[INFO] Skrypt działa strukturalnie poprawnie. Oczekuje zaistnienia rzeczywistych mock-dyskowych danych:")
        print(f" * BRAK IMG: {args.image}")
        print(f" * LUB BRAK WFDB: {args.wfdb}.dat")
