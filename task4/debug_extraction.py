import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Fix relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecg_segmentation import load_checkpoint, predict_mask_array
from ecg_digitization import load_rgb_image, digitize_mask
from ecg_data import load_wfdb_signals
from validate_signals import pearson_corr

def main():
    image_path = Path('../ecg_dataset/train/ecg_train_0005.png')
    hea_path = Path('../ecg_dataset/train/ecg_train_0005.hea')
    checkpoint_path = Path('../artifacts/unet_resnet50.pt')

    if not image_path.exists():
        print(f"Could not find image at {image_path}. Please adjust paths in script.")
        image_path = Path('../../tasks_data/task4/train/ecg_train_0005.png')
        hea_path = Path('../../tasks_data/task4/train/ecg_train_0005.hea')
        checkpoint_path = Path('../../tasks_data/task4/artifacts/unet_resnet50.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, image_size, threshold = load_checkpoint(checkpoint_path, device)
    image_rgb = load_rgb_image(image_path)

    print('Predicting UNet Mask...')
    mask = predict_mask_array(model, image_rgb, image_size=image_size, threshold=threshold, device=device)
    cv2.imwrite('debug_pred_mask_final.png', mask)
    print("Saved 'debug_pred_mask_final.png'. Please check if the traces are HORIZONTAL and clean!")

    print('Extracting Signals from Mask (Viterbi is OFF for UNet)...')
    extracted = digitize_mask(mask, num_samples=1250, image_rgb=image_rgb)
    gt_signals = load_wfdb_signals(hea_path)

    lead_name = 'V2'
    ext = extracted.get(lead_name)

    if ext is None:
        print(f"Extraction failed for lead {lead_name}")
        return

    # V2 is in Col 1, so indices 1250:2500
    gt_full = gt_signals.get(lead_name)
    gt_slice = gt_full[1250:2500]

    # Resample extracted if lengths differ
    src_x = np.linspace(0, 1, len(ext))
    tgt_x = np.linspace(0, 1, len(gt_slice))
    ext_resampled = np.interp(tgt_x, src_x, ext)

    corr = pearson_corr(ext_resampled, gt_slice)
    print(f'\n--- RESULTS ---')
    print(f'Lead {lead_name} Pearson: {corr:.4f}')
    print(f'Extracted Std: {np.std(ext_resampled):.4f}')
    print(f'Ground Truth Std: {np.std(gt_slice):.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(gt_slice, label='Ground Truth', color='blue')
    plt.plot(ext_resampled, label='UNet Extracted', color='red', alpha=0.7)
    plt.title(f'Lead {lead_name} Comparison (Pearson: {corr:.4f})')
    plt.legend()
    plt.savefig('debug_signal_v2.png')
    print('Saved debug_signal_v2.png. Please upload this image so I can see what went wrong with the interpolation!')

if __name__ == '__main__':
    main()
