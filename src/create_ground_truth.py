"""
create_ground_truth.py

Generates ground truth via SAR change detection:
  - Compare Sentinel-1 SAR between two dates
  - Large changes = landslide zones
  - Uses percentile threshold
"""

import numpy as np
import os, glob
import torch.nn.functional as F
import torch

try:
    import rasterio
except:
    os.system("pip install rasterio --break-system-packages -q")
    import rasterio


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data


def resize_arr(arr, h, w):
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (h, w), mode='bilinear',
                         align_corners=False).squeeze().numpy()


def normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def create_ground_truth(base_dir,
                        date1='2024-12-11',
                        date2='2024-12-16',
                        target_h=128, target_w=128,
                        threshold_percentile=70):   # ✅ lowered threshold

    print("="*50)
    print("🛰️ Generating Ground Truth via SAR Change Detection")
    print("="*50)

    # ✅ FIXED SAVE PATH
    save_path = os.path.join(base_dir, 'ground_truth.npy')

    # Load SAR files
    s1_d1 = glob.glob(os.path.join(base_dir, date1, 'Sentinel-1', '*.tif'))
    s1_d2 = glob.glob(os.path.join(base_dir, date2, 'Sentinel-1', '*.tif'))

    # fallback to Sentinel-2 if SAR missing
    if not s1_d1 or not s1_d2:
        print("⚠️ SAR not found, using Sentinel-2 B04")
        s1_d1 = glob.glob(os.path.join(base_dir, date1, 'Sentinel-2', 'B04.tif'))
        s1_d2 = glob.glob(os.path.join(base_dir, date2, 'Sentinel-2', 'B04.tif'))

    print(f"Date1: {os.path.basename(s1_d1[0])}")
    print(f"Date2: {os.path.basename(s1_d2[0])}")

    arr1 = read_tif(s1_d1[0])
    arr2 = read_tif(s1_d2[0])

    arr1 = arr1[0] if arr1.ndim == 3 else arr1
    arr2 = arr2[0] if arr2.ndim == 3 else arr2

    arr1 = resize_arr(normalize(arr1), target_h, target_w)
    arr2 = resize_arr(normalize(arr2), target_h, target_w)

    # change detection
    change_map = np.abs(arr2 - arr1)

    # DEM slope integration
    dem_files = glob.glob(os.path.join(base_dir, date1, 'DEM', '*.tif'))
    if dem_files:
        dem = read_tif(dem_files[0])
        dem = resize_arr(normalize(dem[0]), target_h, target_w)

        dx = np.gradient(dem, axis=1)
        dy = np.gradient(dem, axis=0)
        slope = np.sqrt(dx**2 + dy**2)
        slope = normalize(slope)

        change_map = change_map * (1 + 0.5 * slope)
        print("✅ DEM slope added")

    # ✅ percentile threshold (FIXED)
    thresh = np.percentile(change_map, threshold_percentile)
    binary_mask = (change_map >= thresh).astype(np.float32)

    # ✅ DEBUG PRINT (VERY IMPORTANT)
    print("DEBUG unique labels:", np.unique(binary_mask))

    # save inside dataset folder
    os.makedirs(base_dir, exist_ok=True)
    np.save(save_path, binary_mask)

    n_ls = binary_mask.sum()
    print("\n✅ Ground truth generated!")
    print(f"Shape: {binary_mask.shape}")
    print(f"Landslide pixels: {int(n_ls)} / {binary_mask.size} "
          f"({100*n_ls/binary_mask.size:.2f}%)")
    print(f"Saved at: {save_path}")

    return binary_mask


if __name__ == "__main__":
    create_ground_truth(
        base_dir='data/Wayanad_validation_data'
    )