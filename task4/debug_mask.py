import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add task4 to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from ecg_segmentation import load_checkpoint, predict_mask_array
from ecg_digitization import load_rgb_image
from ecg_data import load_metadata, render_trace_mask

def main():
    image_path = Path('../../tasks_data/task4/train/ecg_train_0005.png')
    json_path = Path('../../tasks_data/task4/train/ecg_train_0005.json')
    checkpoint_path = Path('../../tasks_data/task4/artifacts/unet_resnet50.pt')

    if not checkpoint_path.exists():
        checkpoint_path = Path('../artifacts/unet_resnet50.pt')
    if not image_path.exists():
        image_path = Path('../ecg_dataset/train/ecg_train_0005.png')
        json_path = Path('../ecg_dataset/train/ecg_train_0005.json')

    # 1. Load ground truth mask
    image_rgb = load_rgb_image(image_path)
    metadata = load_metadata(json_path)
    gt_mask = render_trace_mask(metadata, image_rgb.shape[:2], thickness=5)
    
    # 2. Predict UNet mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, image_size, threshold = load_checkpoint(checkpoint_path, device)
    pred_mask = predict_mask_array(model, image_rgb, image_size=image_size, threshold=threshold, device=device)
    
    # 3. Analyze
    gt_nz = np.count_nonzero(gt_mask)
    pred_nz = np.count_nonzero(pred_mask)
    
    intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
    iou = intersection / union if union > 0 else 0.0
    
    print(f"Ground Truth mask nonzero pixels: {gt_nz}")
    print(f"Predicted mask nonzero pixels:    {pred_nz}")
    print(f"Intersection:                     {intersection}")
    print(f"IoU:                              {iou:.4f}")
    
    # Save the predicted mask so the user can look at it
    cv2.imwrite('debug_pred_mask.png', pred_mask)
    cv2.imwrite('debug_gt_mask.png', gt_mask)
    print("Saved debug_pred_mask.png and debug_gt_mask.png")

if __name__ == '__main__':
    main()
