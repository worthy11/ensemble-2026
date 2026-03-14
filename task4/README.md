# Task 4 ECG Digitization Pipeline

This repository uses two dedicated entrypoints:

- `task4/run_pipeline.py` for the deep-learning pipeline
- `task4/run_pipeline_classical.py` for a classical OpenCV baseline (no training)

The script runs all stages in order:

1. generate training masks from `train/*.json`
2. split into train/val
3. train U-Net (ResNet34 encoder, ImageNet pretrained)
4. predict masks for `test/`
5. convert masks into 1D ECG signals
6. save final submission `.npz`

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the full deep-learning pipeline

```bash
python task4/run_pipeline.py
```

## Common deep-learning options

```bash
python task4/run_pipeline.py --epochs 20 --batch-size 16 --lr 2e-4 --threshold 0.55
```

```bash
python task4/run_pipeline.py --limit 120 --epochs 3
```

## Run the classical pipeline (no deep learning)

```bash
python task4/run_pipeline_classical.py
```

```bash
python task4/run_pipeline_classical.py --test-limit 20
```

## Key outputs

- prepared dataset: `data/segmentation_full/`
- trained model: `artifacts/unet_resnet34_full.pt`
- predicted masks: `data/test_masks_full/`
- final 1D submission: `data/out/ecg_example_submission.npz`

The submission archive contains keys in format `record_lead`, e.g. `ecg_test_0001_I`, with `1250` samples per lead stored as `float16`.