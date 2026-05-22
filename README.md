# 🫁 Explainable 3D Liver Tumour Segmentation using U-Net and Grad-CAM

A deep learning system for **volumetric liver tumour segmentation** in contrast-enhanced abdominal CT scans, with integrated **Grad-CAM explainability** to visualise model decision-making. Built using a 3D U-Net architecture trained on the [LiTS (Liver Tumour Segmentation) Challenge dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation).

---

## Overview

This project addresses automatic liver tumour delineation in 3D CT volumes. Unlike most approaches that process CT scans slice-by-slice in 2D, this system operates **fully volumetrically** — extracting 3D patches and learning spatial tumour context across all three dimensions simultaneously.

A key contribution is the integration of **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which produces saliency heatmaps revealing which regions of the CT scan drove the model's prediction — making the system interpretable and clinically trustworthy. A **Gradio web interface** (`app.py`) is included for quick visual inference without any coding.

---

## Model

A custom **3D U-Net** with three encoder levels (32 → 64 → 128 channels), a 256-channel bottleneck, and a symmetric decoder with skip connections. Trained with a composite **Dice + 2×BCE loss** to handle severe class imbalance between tumour and background voxels.

---

## Results

| Threshold | Dice (DSC) | IoU    | Pixel Accuracy |
|-----------|------------|--------|----------------|
| 0.2       | 0.2705     | 0.1970 | 86.61%         |
| 0.3       | 0.2171     | 0.1624 | 89.40%         |
| 0.4       | 0.3321     | 0.2414 | 91.05%         |
| **0.5 ★** | **0.3737** | **0.2899** | **91.55%** |

> Best validation Dice: **0.3952** (epoch 17). Model was trained on 35 patients — performance is expected to improve substantially with a larger cohort.

---

## Dataset

[LiTS Challenge](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) — 51 contrast-enhanced abdominal CT volumes in NIfTI format. Labels: 0 = background, 1 = liver, 2 = tumour. Split 70/15/15 at the patient level.

---

## References

- Bilic, P. et al. (2023). *The Liver Tumor Segmentation Benchmark (LiTS)*. Medical Image Analysis.
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Çiçek, Ö. et al. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. MICCAI.