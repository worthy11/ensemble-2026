import sys
import cv2
import numpy as np

mask_train_path = '../data/segmentation_full/train/masks/ecg_train_0005.png'
mask_val_path = '../data/segmentation_full/val/masks/ecg_train_0005.png'
debug_gt_path = 'debug_gt_mask.png'
pred_mask_path = 'debug_pred_mask.png'

# Find the cached mask
cached_mask_path = mask_train_path
cached_mask = cv2.imread(cached_mask_path, cv2.IMREAD_GRAYSCALE)
if cached_mask is None:
    cached_mask_path = mask_val_path
    cached_mask = cv2.imread(cached_mask_path, cv2.IMREAD_GRAYSCALE)

if cached_mask is None:
    print('ecg_train_0005 mask not found in cached folder.')
    sys.exit(0)
    
debug_mask = cv2.imread(debug_gt_path, cv2.IMREAD_GRAYSCALE)
pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

if debug_mask is None or pred_mask is None:
    print("Run debug_mask.py first to generate debug_gt_mask.png and debug_pred_mask.png")
    sys.exit(0)

print(f'Found cached mask at {cached_mask_path}')
print(f'Cached mask non-zero: {np.count_nonzero(cached_mask)}')
print(f'Debug GT mask non-zero: {np.count_nonzero(debug_mask)}')
print(f'UNet Predicted mask non-zero: {np.count_nonzero(pred_mask)}')

print("\n--- Compare CACHED Training Mask vs FRESH Ground Truth ---")
intersection = np.logical_and(cached_mask > 0, debug_mask > 0).sum()
union = np.logical_or(cached_mask > 0, debug_mask > 0).sum()
print(f'Intersection: {intersection}, Union: {union}')
print(f'IoU: {intersection/union if union > 0 else 0:.4f}')

print("\n--- Compare CACHED Training Mask vs UNET Prediction ---")
inter_pred = np.logical_and(cached_mask > 0, pred_mask > 0).sum()
union_pred = np.logical_or(cached_mask > 0, pred_mask > 0).sum()
print(f'IoU: {inter_pred/union_pred if union_pred > 0 else 0:.4f}')
