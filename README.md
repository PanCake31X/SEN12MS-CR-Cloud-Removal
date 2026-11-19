# SEN12MS-CR Cloud Removal: Two-Stage Latent Mamba Flow Matching & Multimodal Refinement

Official implementation and reproducible paper artifacts for **multimodal SAR-guided optical cloud removal** on the **SEN12MS-CR** dataset using a two-stage **Latent Mamba Conditional Flow Matching** and **Stage-2 Multimodal Refinement** pipeline.

**Author:** Bharat K

---

## Overview

Cloud cover frequently degrades optical satellite imagery (Sentinel-2), limiting downstream Earth observation tasks. Because Synthetic Aperture Radar (Sentinel-1 SAR) penetrates clouds, fusing SAR and cloudy optical observations enables reliable cloud-free reconstruction across both thin haze and optically thick cloud cover.

This repository implements a lightweight, **two-stage cloud removal framework** (~10.25M total parameters, <93 ms full-pipeline latency):

1. **Stage 1 — Latent Mamba Conditional Flow Matching (`src/models/flow_matching.py`):**
   * Uses a compact Variational Autoencoder (**128,308** parameters) paired with a Mamba-based velocity predictor (**5,022,344** parameters) to learn a continuous-time flow field $v_\theta(x_t, t \mid c)$ conditioned on Sentinel-1 SAR and cloudy Sentinel-2 imagery.
   * Produces structurally coherent global reconstructions even under complete thick-cloud occlusion (improving thick-cloud PSNR from $12.73\text{ dB}$ to $27.24\text{ dB}$ in $53.22\text{ ms}$).

2. **Stage 2 — Multimodal Spectral & Detail Refiner (`src/models/refiner.py`, `src/train_stage2_refiner.py`):**
   * A dedicated pixel-space refinement network (**~5.10M** parameters) that fuses four complementary input streams: the Stage-1 reconstruction, the 13-band cloudy Sentinel-2 input, raw dual-pol Sentinel-1 SAR, and $5\times 5$ despeckled SAR.
   * Two variants are benchmarked:
     * **Stage 2 v2 (Fidelity-Focused):** Maximizes pixel/spectral fidelity overall (**32.32 dB PSNR**, **5.31° SAM**, **0.0184 MAE**) and on thin-haze tiles (**34.04 dB PSNR**).
     * **Stage 2 v3 (Perceptual & Thick-Cloud Specialist):** Achieves the best perceptual quality overall (**0.2341 LPIPS**, **0.9446 SSIM**) and leads on thick-cloud reconstruction across SSIM (**0.9220**), SAM (**6.89°**), MAE (**0.0208**), RMSE (**0.0329**), and LPIPS (**0.3321**).

---

## Repository Structure

```text
SEN12MS-CR-Cloud-Removal/
├── notebooks/
│   └── MI_X16.ipynb                # End-to-end Kaggle training & evaluation notebook
├── src/
│   ├── models/
│   │   ├── flow_matching.py        # Stage-1 Latent Mamba Conditional Flow Matching model
│   │   ├── refiner.py              # Stage-2 Multimodal Refiner architecture (v2 / v3)
│   │   └── inference.py            # Unified ODE sampling & two-stage inference pipeline
│   └── train_stage2_refiner.py     # Stage-2 Refiner training script
├── eval_manifest.json              # Fixed 100-tile multi-season evaluation manifest
├── test_summary.json               # Aggregated metrics, latency, params & weight audit
├── weight_audit.json               # Input-branch first-layer weight norm audit
└── README.md
```

---

## Evaluation Protocol

Evaluation is conducted on a fixed, reproducible **100-tile multi-season benchmark** sourced from `Hermanni/sen12mscr` (5 shards $\times$ 20 tiles each, defined in `eval_manifest.json`):

* **Shards:** `fall/scene_120.parquet`, `fall/scene_65.parquet`, `spring/scene_39.parquet`, `summer/scene_139.parquet`, `summer/scene_86.parquet`
* **Regime Stratification:**
  * **All Tiles:** $n = 100$
  * **Thin Cloud / Haze:** $n = 53$
  * **Thick Cloud:** $n = 47$

---

## Quantitative Results

All metrics are reported as $\text{mean} \pm \text{std}$ across the 100-tile SEN12MS-CR evaluation set (`test_summary.json`). Best results per metric are highlighted in **bold**.

### Overall Benchmark ($n = 100$)

| Method | PSNR ($\uparrow$) | SSIM ($\uparrow$) | SAM ($\downarrow$) | MAE ($\downarrow$) | RMSE ($\downarrow$) | LPIPS ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Cloudy Input | $19.99 \pm 8.39$ | $0.6608 \pm 0.2307$ | $10.86 \pm 6.90$ | $0.1103 \pm 0.0937$ | $0.1458 \pm 0.1100$ | $0.5027 \pm 0.2526$ |
| Stage 1 (Flow Matching) | $27.18 \pm 2.26$ | $0.8833 \pm 0.0542$ | $10.10 \pm 2.87$ | $0.0334 \pm 0.0101$ | $0.0453 \pm 0.0119$ | $0.6698 \pm 0.1445$ |
| **Stage 1 + Stage 2 v2** | **$32.32 \pm 3.58$** | $0.9420 \pm 0.0512$ | **$5.31 \pm 3.92$** | **$0.0184 \pm 0.0099$** | **$0.0265 \pm 0.0125$** | $0.2660 \pm 0.1767$ |
| **Stage 1 + Stage 2 v3** | $31.16 \pm 3.76$ | **$0.9446 \pm 0.0478$** | $5.38 \pm 3.16$ | $0.0203 \pm 0.0116$ | $0.0303 \pm 0.0135$ | **$0.2341 \pm 0.1637$** |

### Thin Cloud / Haze Regime ($n = 53$)

| Method | PSNR ($\uparrow$) | SSIM ($\uparrow$) | SAM ($\downarrow$) | MAE ($\downarrow$) | RMSE ($\downarrow$) | LPIPS ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Cloudy Input | $26.44 \pm 6.17$ | $0.8455 \pm 0.1163$ | $6.86 \pm 5.25$ | $0.0415 \pm 0.0240$ | $0.0597 \pm 0.0359$ | $0.3474 \pm 0.2440$ |
| Stage 1 (Flow Matching) | $27.13 \pm 2.20$ | $0.8943 \pm 0.0462$ | $9.71 \pm 2.42$ | $0.0336 \pm 0.0092$ | $0.0454 \pm 0.0114$ | $0.6567 \pm 0.1398$ |
| **Stage 1 + Stage 2 v2** | **$34.04 \pm 2.41$** | **$0.9692 \pm 0.0176$** | **$3.39 \pm 1.44$** | **$0.0137 \pm 0.0032$** | **$0.0206 \pm 0.0057$** | $0.1683 \pm 0.0849$ |
| **Stage 1 + Stage 2 v3** | $32.25 \pm 4.48$ | $0.9646 \pm 0.0253$ | $4.04 \pm 2.35$ | $0.0198 \pm 0.0149$ | $0.0281 \pm 0.0164$ | **$0.1472 \pm 0.1002$** |

### Thick Cloud Regime ($n = 47$)

| Method | PSNR ($\uparrow$) | SSIM ($\uparrow$) | SAM ($\downarrow$) | MAE ($\downarrow$) | RMSE ($\downarrow$) | LPIPS ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Cloudy Input | $12.73 \pm 2.70$ | $0.4524 \pm 0.1271$ | $15.36 \pm 5.65$ | $0.1879 \pm 0.0817$ | $0.2428 \pm 0.0806$ | $0.6777 \pm 0.1038$ |
| Stage 1 (Flow Matching) | $27.24 \pm 2.33$ | $0.8708 \pm 0.0596$ | $10.54 \pm 3.25$ | $0.0333 \pm 0.0111$ | $0.0451 \pm 0.0125$ | $0.6846 \pm 0.1482$ |
| **Stage 1 + Stage 2 v2** | **$30.37 \pm 3.68$** | $0.9115 \pm 0.0589$ | $7.46 \pm 4.65$ | $0.0237 \pm 0.0120$ | $0.0332 \pm 0.0146$ | $0.3761 \pm 0.1882$ |
| **Stage 1 + Stage 2 v3** | $29.94 \pm 2.16$ | **$0.9220 \pm 0.0563$** | **$6.89 \pm 3.27$** | **$0.0208 \pm 0.0060$** | **$0.0329 \pm 0.0087$** | **$0.3321 \pm 0.1661$** |

---

## Model Complexity, Latency & Multimodal Weight Audit

### Parameters & Inference Latency

| Component / Pipeline | Parameters | Latency (ms / tile) |
| :--- | :---: | :---: |
| Stage 1 VAE | 128,308 | — |
| Stage 1 Mamba Predictor | 5,022,344 | — |
| **Stage 1 Total** | **5,150,652** | **53.22 ms** |
| Stage 2 v2 Refiner | 5,097,797 | 36.12 ms |
| Stage 2 v3 Refiner | 5,097,993 | 39.16 ms |
| **Full Pipeline (Stage 1 + v2)** | **10,248,449** | **89.34 ms** |
| **Full Pipeline (Stage 1 + v3)** | **10,248,645** | **92.38 ms** |

### Input-Branch Weight Audit (`weight_audit.json`)

To verify genuine multimodal fusion (and rule out SAR branch collapse), we audit the mean input-layer weight norms across all four Stage-2 input branches:

| Refiner Variant | Stage-1 Pred | Optical 13-Band | Raw SAR | Despeckled $5\times 5$ SAR | Max Weight Mag. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Stage 2 v2 | $0.9201$ | $0.8957$ | $0.8189$ | $0.8695$ | $1.9716$ |
| Stage 2 v3 | $0.9329$ | $0.9034$ | $0.8143$ | $0.8705$ | $1.9724$ |

Both refiners maintain balanced weight norms across all four modalities, consistently assigning higher weight norm to $5\times 5$ despeckled SAR than raw SAR.

---

## Installation & Usage

```bash
git clone https://github.com/<YOUR_USERNAME>/SEN12MS-CR-Cloud-Removal.git
cd SEN12MS-CR-Cloud-Removal

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy tqdm matplotlib rasterio einops scikit-image lpips datasets pyarrow
```

* **Full Notebook Reproduction:** Run `notebooks/MI_X16.ipynb`.
* **Train Stage-2 Refiner:** `python src/train_stage2_refiner.py`
* **Run Two-Stage Inference:** `python src/models/inference.py`
* **Paper Artifacts Tag:** `git checkout tags/v1.0-paper-artifacts`

---

## Citation

If you find this code or benchmark useful in your research, please cite:

```bibtex
@article{bharatk2026sen12mscr,
  author  = {Bharat K},
  title   = {Two-Stage Latent Mamba Flow Matching and Multimodal Refinement for SAR-Guided Optical Cloud Removal},
  journal = {IEEE Transactions on Geoscience and Remote Sensing (Preprint)},
  year    = {2026}
}
```

Please also cite the original SEN12MS-CR dataset:

```bibtex
@article{ebel2020multisensor,
  author  = {Ebel, Patrick and Meraner, Andrea and Schmitt, Michael and Zhu, Xiao Xiang},
  title   = {Multisensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {59},
  number  = {7},
  pages   = {5866--5878},
  year    = {2021},
  doi     = {10.1109/TGRS.2020.3024744}
}
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.