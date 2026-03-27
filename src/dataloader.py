"""
dataloader.py — Wayanad landslide data loader
Ground truth = generated via SAR change detection

FIXES applied:
  1. Mismatched channel counts across dates (date-2 has no Rainfall band).
     → Pad missing bands with zeros so every date has the same C channels.
  2. Boundary off-by-one: loop used range(half, H-half, stride) but patch
     i+half could equal H, producing a truncated slice that failed the
     shape check and silently dropped every patch.
     → Use range(0, H-patch_size+1, stride) — standard sliding-window idiom.
"""

import os, glob, zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import rasterio
except Exception:
    os.system("pip install rasterio --break-system-packages -q")
    import rasterio

try:
    import xarray as xr
except Exception:
    os.system("pip install xarray netCDF4 --break-system-packages -q")
    import xarray as xr


# ──────────────────────────────────────────────
#  Low-level I/O helpers
# ──────────────────────────────────────────────

def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data


def read_nc(path):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    arr = ds[var].values.astype(np.float32)
    return arr.mean(axis=0) if arr.ndim == 3 else arr


def extract_zip(zpath, out):
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(out)
    for f in os.listdir(out):
        if f.endswith('.tif'):
            return os.path.join(out, f)
    return None


def normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def resize_band(arr, h, w):
    import torch.nn.functional as F
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (h, w), mode='bilinear', align_corners=False).squeeze().numpy()


# ──────────────────────────────────────────────
#  Per-date feature loader
# ──────────────────────────────────────────────

def load_one_date(folder, target_h=128, target_w=128):
    bands, names = [], []
    print(f"\n📂 {os.path.basename(folder)}")

    # DEM
    p = os.path.join(folder, 'DEM', 'Copernicus_DEM_30m.tif')
    if os.path.exists(p):
        bands.append(resize_band(normalize(read_tif(p)[0]), target_h, target_w))
        names.append('DEM')
        print("  ✅ DEM")

    # Sentinel-1
    s1 = glob.glob(os.path.join(folder, 'Sentinel-1', '*.tif'))
    if s1:
        arr = read_tif(s1[0])
        for b in range(min(arr.shape[0], 2)):
            bands.append(resize_band(normalize(arr[b]), target_h, target_w))
            names.append(f'SAR_B{b+1}')
        print(f"  ✅ SAR ({arr.shape[0]} bands)")

    # Sentinel-2
    s2d = os.path.join(folder, 'Sentinel-2')
    c = 0
    for b in ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']:
        p = os.path.join(s2d, f'{b}.tif')
        if os.path.exists(p):
            bands.append(resize_band(normalize(read_tif(p)[0]), target_h, target_w))
            names.append(f'S2_{b}')
            c += 1
    if c:
        print(f"  ✅ Sentinel-2 ({c} bands)")

    # Rainfall
    p = os.path.join(folder, 'Rainfall Data', 'kerala_rainfall_data.nc')
    if os.path.exists(p):
        bands.append(resize_band(normalize(read_nc(p)), target_h, target_w))
        names.append('Rainfall')
        print("  ✅ Rainfall")
    else:
        # ── FIX 1 ── Missing rainfall → zero-filled placeholder band
        # This ensures all dates have the same number of channels.
        bands.append(np.zeros((target_h, target_w), dtype=np.float32))
        names.append('Rainfall_zero')
        print("  ⚠️  Rainfall missing — zero band added to keep channel count consistent")

    features = np.stack(bands, axis=0)
    print(f"  📊 Shape:{features.shape} | Bands:{names}")
    return features, names


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class LandslideDataset(Dataset):
    def __init__(self, features_list, labels, patch_size=16, stride=8):
        """
        features_list : list of (C, H, W) arrays — one per date/timestep
        labels        : (H, W) binary array from ground truth
        patch_size    : spatial size of each patch (pixels)
        stride        : sliding-window stride
        """
        self.patches, self.targets = [], []

        T = len(features_list)
        C, H, W = features_list[0].shape

        # ── FIX 2 ── Standard sliding-window: top-left corner goes from
        # 0 to (H - patch_size) inclusive, so i+patch_size <= H always.
        # The old range(half, H-half, stride) could yield i+half == H,
        # producing a truncated slice that never matched (C, P, P).
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):

                pts = [
                    features_list[t][:, i:i + patch_size, j:j + patch_size]
                    for t in range(T)
                ]

                # Sanity check — should always pass now
                if not all(p.shape == (C, patch_size, patch_size) for p in pts):
                    continue

                self.patches.append(np.stack(pts, axis=0))   # (T, C, P, P)
                self.targets.append(int(labels[i + patch_size // 2,
                                               j + patch_size // 2]))

        self.patches = np.array(self.patches, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

        n = int(self.targets.sum())
        print(f"✅ {len(self.patches)} patches | "
              f"Landslide:{n} | No-slide:{len(self.targets) - n}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.patches[idx]),
                torch.tensor(self.targets[idx]))


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def get_dataloaders(base_dir,
                    date_folders=('2024-12-11', '2024-12-16'),
                    target_h=128, target_w=128,
                    patch_size=16, stride=8,
                    batch_size=32):

    from create_ground_truth import create_ground_truth

    # Ground truth
    gt_path = os.path.join(base_dir, 'ground_truth.npy')
    if os.path.exists(gt_path):
        print("✅ Loading existing ground truth...")
        atlas = np.load(gt_path)
    else:
        print("⚡ Generating ground truth...")
        atlas = create_ground_truth(
            base_dir, date_folders[0], date_folders[1],
            target_h, target_w
        )

    # Load features for all dates
    all_features, band_names = [], []
    for date in date_folders:
        feats, names = load_one_date(
            os.path.join(base_dir, date), target_h, target_w
        )
        all_features.append(feats)
        band_names = names   # use the last date's names (they match after padding)

    # Verify channel consistency
    shapes = [f.shape for f in all_features]
    if len(set(s[0] for s in shapes)) > 1:
        raise ValueError(
            f"Channel mismatch across dates even after padding: {shapes}\n"
            "Check that load_one_date pads every date to the same band list."
        )

    # Build dataset
    full_ds = LandslideDataset(all_features, atlas, patch_size, stride)

    if len(full_ds) == 0:
        raise RuntimeError(
            "Dataset is empty after patch extraction.\n"
            f"  features shape : {all_features[0].shape}\n"
            f"  labels shape   : {atlas.shape}\n"
            f"  patch_size={patch_size}, stride={stride}\n"
            "Ensure patch_size < min(H, W) of the rasters."
        )

    labels = full_ds.targets
    idx    = list(range(len(full_ds)))

    # Stratified split: 70 / 10 / 20
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=42
    )
    tr_idx, va_idx = train_test_split(
        tr_idx, test_size=0.125,
        stratify=labels[tr_idx], random_state=42
    )

    from torch.utils.data import Subset

    tl  = DataLoader(Subset(full_ds, tr_idx), batch_size=batch_size, shuffle=True,  drop_last=True)
    vl  = DataLoader(Subset(full_ds, va_idx), batch_size=batch_size, shuffle=False)
    tsl = DataLoader(Subset(full_ds, te_idx), batch_size=batch_size, shuffle=False)

    print(f"\n✅ Train:{len(tr_idx)} | Val:{len(va_idx)} | Test:{len(te_idx)}")

    num_channels  = all_features[0].shape[0]
    num_timesteps = len(date_folders)
    return tl, vl, tsl, num_channels, num_timesteps, band_names
