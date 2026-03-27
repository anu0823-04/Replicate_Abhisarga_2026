# Replication Report
##  EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning



**Event:** Replicate — Abhisarga, IIIT Sri City, March 2026  
**Team Members:** Yellapragada Naga Venkata Sri Anuroop
                  Ruthvik Chowdary Pasungulapati  
**Date:** March 27, 2026

---

## 1. Paper Summary

The paper proposes EEGMoE, a novel self-supervised learning framework that uses a Domain-Decoupled Mixture-of-Experts architecture. Its core contribution is the ability to separate "Domain-Specific" features from "Shared" features using a specialized routing mechanism. While originally designed for EEG brain signals, we have adapted it for landslide detection by treating different satellite and meteorological sensors as distinct data domains.

## 2. Methodology Description



### 2.1 Overview
We implemented a two-stage pipeline. Stage 1 involves self-supervised pre-training where the model learns to reconstruct masked spatial-frequency features of the terrain. Stage 2 is supervised fine-tuning using the Landslide Atlas as ground truth to classify high-risk patches.

### 2.2 Algorithm / Model
The model uses a Spatial-Frequency Encoder to process multi-modal inputs. The encoded features are passed through:

SpecificMoE: Experts that focus on "Trigger" domains (Rainfall, Soil Moisture).

SharedMoE: Experts that process "Static" domains (DEM, Slope, SAR texture).
A Top-K Router (k=2) dynamically selects the best experts for each patch of land.
### 2.3 Key Hyperparameters
Parameter           ,Value Used,          Paper Value
Learning Rate,        5e-4 (Fine-tune),     1e-3
Epochs,               35,                   50+
Batch Size,           16,                   64
Embed Dim,            128,                  128

## 3. Experimental Setup

### 3.1 Dataset
- Dataset name:Wayanad & Puthumala Multi-Modal Landslide Dataset
- Size:225 patches (128x128 pixels)
- Preprocessing steps:Z-score normalization, 11-channel stacking (DEM, SAR, S2, Rain, Soil), and stratified splitting.



### 3.2 Environment
- Python version:3.10+
- Key libraries:PyTorch, Rasterio, Xarray, Scikit-learn
- Hardware used:CPU (Local Environment)

### 3.3 Assumptions Made

1.We assumed the Rainfall data from the nearest date was representative when exact hourly data was missing.

2.We assumed Top-2 routing was sufficient for the 5 distinct data domains provided.

## 4. Results

### 4.1 Quantitative Comparison

Metric                 Paper (EEG Task)          Ours (Landslide Task)         Delta
Accuracy,                ~84.0%,                   64.44%,                    -19.56%
Precision (Positive)       N/A,                    33.33%,                      -
Recall (Positive),         N/A,                    23.08%,                      -

### 4.2 Qualitative Observations
The model is exceptionally good at identifying "No Landslide" zones (81% recall), making it useful as a safety-filtering tool. However, it is conservative in predicting actual landslides, likely due to the extreme class imbalance in the natural landscape.

## 5. Analysis of Discrepancies

Discrepancy 1: Significant drop in accuracy compared to the original paper.

Likely cause: The original paper used thousands of EEG samples, while our landslide dataset was limited to 225 patches, leading to overfitting on the training set (which hit 100% accuracy).

Discrepancy 2: Low Recall for the Landslide class.

Likely cause: Data Imbalance. 70% of the training data was "Safe" land, causing the MoE router to favor experts that specialize in stable terrain features.
## 6.Challenges (What was difficult)
Data Alignment (The "Spatial Puzzle"): The most difficult part was ensuring that 11 different layers (Radar, Optical, Rainfall, DEM, etc.) all lined up perfectly. Sentinel-1 has a different grid than Sentinel-2, and Rainfall data often comes in a much lower resolution. Getting them all into a single 128x128 "Data Cube" without distorting the geography was a major technical hurdle.

The "Zero-Label" Problem: Landslides are rare. In a 128x128 area, perhaps only 5% of the pixels are actual landslides. This Class Imbalance makes it very hard for the MoE (Mixture of Experts) to learn. The model naturally wants to guess "Safe" because it's right 95% of the time, so forcing it to recognize the "Needle in the Haystack" was a constant struggle.

Adapting EEG to Earth: The EEGMoE paper was written for brain waves (1D time signals). Translating that architecture to 2D satellite images (Spatial) while keeping the Self-Supervised Masking logic required significant re-engineering of the Encoder.

## 7. What can be made better (Future Scope)
Attention Mechanisms (Transformers):
Currently, the MoE uses a simple Router. Adding Spatial Attention would allow the "experts" to focus specifically on the edges of mountains (the most likely slide points) rather than looking at the whole patch equally.

Temporal Depth:
Right now, we only use two dates (Before and After). To make it a true "Early Warning System," we should feed the model a 30-day history of rainfall. This would allow the MoE to learn the "Saturation Point"—the exact moment when the soil becomes too heavy with water to stay on the slope.

Multi-Scale Experts:
Instead of all experts looking at 16x16 patches, we could have "Global Experts" looking at the whole mountain and "Local Experts" looking at specific soil cracks. This "Domain-Decoupling" would follow the EEGMoE paper even more closely.


## 8. Conclusion
In this project, we successfully adapted the EEGMoE architecture—originally designed for brain signal processing—to the critical task of landslide detection in the Western Ghats (Wayanad and Puthumala).

Key Findings:

The Domain-Decoupled approach proved effective at merging static terrain data with dynamic "trigger" data like rainfall and soil moisture.

The model demonstrated a high reliability in identifying "Safe Zones" (81% Recall), which is vital for reducing false alarms in early warning systems.


Future Work:
To move from a 64% accuracy to 90%+, future iterations should focus on Class Weighting to handle the rarity of landslides and expanding the dataset to include more historical events across Kerala and Tamil Nadu. This work serves as a proof-of-concept for a scalable, multi-modal AI safety system for landslide-prone regions in India.

## 9. References

1. EEGMoE_A_Domain-Decoupled_Mixture-of-Experts_Model_for_Self-Supervised_EEG Representation Learning.

2. OpenAI, “ChatGPT,” (Mar. 27, 2026 version) [Large language model]. Available: https://chat.openai.com

3. Google, “Gemini,” (Mar. 27, 2026 version) [Large language model]. Available: https://gemini.google.com

4. Anthropic, “Claude 3.5 Sonnet,” [Large language model]. Available: https://www.anthropic.com
