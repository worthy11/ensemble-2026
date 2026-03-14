# Instrukcja Sieci Dotter (Digitalizacja EKG)

Poniżej znajduje się opis stworzonego szkieletu modelu *Dotter* słuzącego do segmentacji siatki EKG na zaszumionych obrazach, tworzącego ostatecznie binarne, czarno-białe maski detekujące główne punkty przedłużenia/przecięcia.

## Pliki Projektu
- `dotter_model.py` - Implementacja właściwa architektury U-Net połączonej z ResBlocks, tworząc model Dottera służący do dygitalizacji/segmentacji. Kod wspiera operacje na tenzorach CUDA i oparty jest bezpośrednio na PyTorch by dawać najwyższą moc obliczeń. 
- `dataset.py` - Dedykowany moduł Dataset dla PyTorch umożliwiający odczyt danych wejściowych z OpenCV, a następnie formatowanie w celach predykcyjnych sieci (C,H,W - RGB tensor) w przedziale pikselowym 0.0-1.0. 
- `inference.py` - Uruchamialny skrypt przeznaczony do wykonania i testowania na paczkach z pacjentami. Skrypt przetwarza foldery ze zdjęciami, przekazuje zoptymalizowanym DataLoadem dane z karty graficznej Nvidia Cuda, przesyłając i wyciągając segmentacje ostatecznie w formacie zwykłego twardego obrazu *.png* *.jpg*.

## Zmiana ustawień / konfigurowanie ścieżek
Podczas działania, potrzebujesz by folder inferencji zgadzał się w pliku `inference.py`. Wystarczy przejść do pierwszych kilkunastu wierszy tegoż pliku.

### Krok po kroku:
0. Zainstaluj wymagane biblioteki, w tym `segmentation-models-pytorch` (która dostarcza gotową architekturę z encoderem resnet34):
   ```bash
   pip install torch torchvision opencv-python segmentation-models-pytorch
   ```
1. Otwórz u siebie plik `inference.py`.
2. Odszukaj sekcji na samej jego górze pod importami, oznaczonych jako `KONFIGURACJA ŚCIEŻEK`.

```python
INPUT_IMAGE_DIR = "TWOJA_SCIEZKA_LADOWANIA"
OUTPUT_MASK_DIR = "TWOJA_SCIEZKA_ZUKONCZENIA"
```

3. Skopiuj tu i **wklej swoją właściwą ścieżkę z folderem obrazów testowych** (np. `images/ekg_noises_input`) w zmienną `INPUT_IMAGE_DIR`.
4. Przy zmianie parametru `OUTPUT_MASK_DIR`, decydujesz, na dysku w jakim folderze zadeklarują się wydestylowane obrazy.
5. Po zapisaniu możesz w swoim terminalu uruchomić `python inference.py`.

Dzięki poprawnemu kodowi w architekturze `pin_memory` w przypadku używania karty NVIDIA skrypt będzie używał szyn pamięci z dużą przepustowością by zachowywać optymalny narzut odświeżania cykli.
