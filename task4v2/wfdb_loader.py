import wfdb
import numpy as np

def load_ground_truth(record_path):
    """
    Wczytuje oryginalne (prawdziwe) referencyjne sygnały EKG z plików WFDB 
    (wymagane pliki: .dat i .hea bez rozszerzenia w argumencie).
    
    Zwraca:
    - Słownik { 'nazwa_odprowadzenia': np.array(sygnał_w_mV) }
    
    Biblioteka wfdb automatycznie aplikuje informacje z plików .hea tzn.
    Gain oraz Baseline i zwraca docelowe sygnały w jednostkach fizycznych (mV).
    """
    try:
        # wfdb.rdsamp zwraca tuple (sygnały_jako_numpy_array, metadane_jako_słownik)
        signals, fields = wfdb.rdsamp(record_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Nie znaleziono pliku rekordu {record_path}. Upewnij się, że pliki .dat i .hea istnieją.")
        
    lead_names = fields.get('sig_name', [])
    sampling_rate = fields.get('fs')
    
    if sampling_rate != 500:
        print(f"[WARN] Plik GT '{record_path}' ma próbkowanie {sampling_rate} Hz (oczekiwane 500 Hz).")
        
    gt_dict = {}
    # signals to macierz o wymiarach (liczba_próbek, liczba_odprowadzeń) 
    # Mamy 12 odprowadzeń i typowo 5000 próbek (10 sek).
    for idx, lead in enumerate(lead_names):
        # Pobierz całą kolumnę dla konkretnego odprowadzenia i zrzutuj na docelową jakość
        # Wartości w 'signals' są już przekonwertowane z ADC (Raw) na fizyczne napięcie mV!
        signal_mv = signals[:, idx].astype(np.float16)
        gt_dict[lead] = signal_mv
        
    return gt_dict

if __name__ == "__main__":
    # Testowy uruchamiacz do debugowania czy instalacja wfdb jest poprawna
    try:
        print("[TEST] Próba wczytania wfdb.rdsamp(...)")
        # gt = load_ground_truth("data/gt/sample_record_001")
        print("[TEST] WFDB zainstalowane poprawnie i gotowe do akcji dla Ground Truth.")
    except Exception as e:
        print(f"Błąd we wczytywaniu: {e}")
