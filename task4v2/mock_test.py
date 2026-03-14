import numpy as np
import os

def test_mock_submission():
    """
    Funkcja testująca generująca zmyślone (mockowe) dane analogiczne do predykcji
    i wyrzucająca spłaszczoną matrycę NPZ dla weryfikacji formatu do serwera.
    """
    # Załóżmy 2 wirtualne obrazy-rekordy, każdy sygnał ma długość np. 5000 punktów (10 sec @500Hz) 
    records = ["ecg_mock_001", "ecg_mock_002"]
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    submission_dict = {}
    
    print("[MOCK] Generowanie sygnału 500Hz (float16) dla testowego obiektu...")
    for record in records:
        for lead in leads:
            # Tworzymy wektor 5000 losowych punktów oznaczjących szum EKG
            mock_signal = np.random.randn(5000).astype(np.float16)
            
            # Kluczowanie formatu płaskiego wg wymagań
            flat_key = f"{record}_{lead}"
            
            submission_dict[flat_key] = mock_signal
            
    # Oszczędność wagi na formacie:
    out_file = "data/out/mock_submission.npz"
    os.makedirs("data/out", exist_ok=True)
    np.savez(out_file, **submission_dict)
    
    print(f"[MOCK] Plik testowy zapisany do {out_file}.")
    print("[MOCK] Weryfikacja struktury otwarcia...")
    
    # Odczyt weryfikujący z zapisanego w locie pliku z powrotem do slownika numpy
    loaded_data = np.load(out_file)
    print(f"Zapisane klucze to (razem {len(loaded_data.files)} kluczy). Próbka:")
    for key in loaded_data.files[:5]:
        signal = loaded_data[key]
        print(f" -> Klucz: '{key}' | Typ: {signal.dtype} | Długość/Wymiar 1D: {signal.shape}")

if __name__ == "__main__":
    test_mock_submission()
