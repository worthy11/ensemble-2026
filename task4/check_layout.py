import json
import sys
import numpy as np

json_path = '../../tasks_data/task4/train/ecg_train_0005.json'
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    json_path = '../ecg_dataset/train/ecg_train_0005.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

print(f"Loaded {json_path}")
for lead_info in data.get('leads', []):
    name = lead_info.get('lead_name', '')
    pixels = lead_info.get('plotted_pixels', [])
    if not pixels:
        continue
    arr = np.array(pixels)
    if arr.ndim != 2:
        continue
    r1_min, r1_max = arr[:, 0].min(), arr[:, 0].max()
    r2_min, r2_max = arr[:, 1].min(), arr[:, 1].max()
    print(f"{name:5}: Col 0 [{r1_min:6.1f}, {r1_max:6.1f}] spans {r1_max-r1_min:6.1f} | Col 1 [{r2_min:6.1f}, {r2_max:6.1f}] spans {r2_max-r2_min:6.1f}")
