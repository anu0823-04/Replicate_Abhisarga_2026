# 📄 [Paper Title] — Replication Study

> **Replicate Challenge** | Abhisarga, IIIT Sri City | March 2026

## 👥 Team
| Name                                 | College          

| Yellapragada Naga Venkata Sri Anuroop| IIIT SRICITY
| Ruthvik Chowdary Pasungulapati       | IIIT SRICITY

---

## 📌 Paper Details
-Title: EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning
- **Paper Link:** https://docs.google.com/document/d/13dgLyrD4XiTnSpqbDtiD_KZNaMOtIvhhjOhouJVgEIc/edit?tab=t.0

---

## 🗂️ Repository Structure

```
├── src/                  # Source code
│   ├── model.py          # Domain-Decoupled MoE implementation
│   ├── dataloader.py     # 11-channel data cube generator (SAR, S2, Soil, etc.)
│   ├── train.py          # Two-stage training (Pre-train & Fine-tune)
│   ├── create_ground_truth.py # ISRO Atlas integration logic
│   └── utils.py          # Seed setting and logging
├── data/                 # Wayanad & Puthumala event data folders
├── results/              # Best model weights (.pt) and training logs
├── replication_report.md # Full technical analysis
└── requirements.txt      # Python dependencies

---


# Install dependencies
pip install torch rasterio xarray netCDF4 scikit-learn numpy
📦 Dataset
This project utilizes multi-modal data for the Wayanad (2024) and Puthumala (2019) landslide events.

Sentinel-1 (SAR): C-band radar for terrain texture.

Sentinel-2 (MSI): Optical bands for vegetation indices.

Hydromet: Rainfall (.nc) and Soil Moisture (.tif) trigger factors.


# Reproducing the Results

# Step 1: Generate Ground Truth from Atlas
# Ensure 'atlas_mask.tif' is in your data folder
python src/create_ground_truth.py --base_dir data/Wayanad_validation_data

# Step 2: Train & Evaluate (Two-Stage EEGMoE)
python main.py --base_dir data/Wayanad_validation_data --batch_size 16

---

## 📊 Results Comparison

Metric          Paper Reports (EEG)             Our Reproduction (Landslide)    Difference
Accuracy,         ~84.0%,                            64.44%,                    -19.56%
Recall (Safe)     N/A                                23%,                          -
Precision,       High,                               33.33%,                 Data Imbalance

## 🔍 Key Observations

- *What matched well: The Domain-Decoupling logic successfully separated "Trigger" factors (Rain/Soil) from "Static" factors (Slope), allowing the MoE router to select specialized experts.

Discrepancies found: Accuracy was lower than the original paper's brain-signal tasks.

Likely reasons: 1. Data Imbalance: Landslides are rare "needle-in-haystack" events compared to continuous EEG signals.
2. Spatial Alignment: Resampling satellite data (10m) to match rainfall data (variable) introduces noise.
3. Dataset Size: The model was trained on 225 patches, while the paper utilized significantly larger time-series datasets. ...

---



## 📜 License
This replication study is submitted as part of the Replicate competition at Abhisarga 2026.
"# Replicate_Abhisarga_2026" 
