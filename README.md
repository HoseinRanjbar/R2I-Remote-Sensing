# Interpretable & Robust AI for Remote Sensing — Scene/Object Classification

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://drive.google.com/file/d/12FdN3W_TRrUtb1t0qB0qAq7nW5wP9mhD/view?usp=sharing)

**This repo contains the remote-sensing experiments from our paper:** we transfer robust CNN backbones (CIFAR-100 initialization) to geospatial imagery, fine-tune a linear head, and study how adversarial robustness relates to post-hoc interpretability—showing a smaller accuracy drop under ℓ∞-PGD and more object-aligned attributions than standard models.

## What’s in this repo
- **Task:** Remote-sensing scene/object classification on satellite/aerial imagery (**see `data/README.md` for dataset setup and splits**).
- **Backbone transfer:** RobustBench CIFAR-100 checkpoints; **freeze the backbone** and fine-tune a **linear head** for the RS task.
- **Threat model:** ℓ∞ **PGD** (ε = 4/255, 10 steps, step size 1/255, random start).
- **Explanations:** Saliency, DeepLIFT, Integrated Gradients (via Captum).
- **Attribution coverage:** fraction of attribution mass inside ground-truth **object bounding boxes**.

## Models evaluated (remote sensing)
- *Data Filtering for Efficient Adversarial Training* (Chen et al., 2024) 📄 [paper](https://www.sciencedirect.com/science/article/pii/S0031320324001456) | **WideResNet-34-10**
- *Helper-based Adversarial Training* (Rade & Moosavi-Dezfooli, ICLR 2022) 📄 [paper](https://openreview.net/forum?id=Azh9QBQ4tR7) | **PreActResNet-18**
- *Fixing Data Augmentation to Improve Adversarial Robustness* (Rebuffi et al., 2021) 📄 [paper](https://arxiv.org/abs/2103.01946) | **WideResNet-70-16**
- *Standard* (non-robust baseline) [paper](https://arxiv.org/abs/2010.09670) | **WRN-28-10**

> **Implementation note:** We wrap the RobustBench models and train only the linear head for the RS dataset(s).

## Datasets
- **Remote-sensing dataset(s) with bounding boxes** for objects of interest (e.g., aircraft, ships, vehicles).  
  Use `scripts/prepare_remote_sensing.py` to download/prepare data and generate **train/val/test** splits.  
  Bounding boxes are used to compute **attribution coverage**.

## Training & Evaluation
- **Optimization:** Adam (lr = 1e-3), batch size = 8 (adjust for GPU), up to 25 epochs with early stopping (patience = 5).
- **Frozen features:** only the final linear layer is trained.
- **Robustness:** report **clean** and **PGD** accuracy and **ΔAcc = clean − PGD**.
- **Interpretability:** generate per-image attributions (Saliency / DeepLIFT / Integrated Gradients) and compute **coverage** (% attribution mass inside the ground-truth boxes).

## Results
**Table — Remote Sensing Results.** Test accuracy, adversarial accuracy (PGD, ℓ∞ ε=4/255), accuracy drop (ΔAcc), and links to fine-tuned checkpoints (ckpts).

| Model | Performance Rank | RobustBench Rank | Test Acc (%) | PGD (4/255) (%) | Δ Acc (%) | Fine-tuned Checkpoint (ckpt) |
|---|---:|---:|---:|---:|---:|:--:|
| Chen et al.   | 3 | 38 | 57.33 | 43.00 | 14.33 | [ckpt](https://huggingface.co/MohammadFazli/xAI-remote-sensing-dior-model/blob/main/best_data_filtering_wrn_34_20.pth) |
| Rade et al.    | 1 | 21 | 67.67 | 47.33 | 20.34 | [ckpt](https://huggingface.co/HosseinRanjbar/remote_sensing/blob/main/best_helper_remote_sensing.pth) |
| Rebuffi et al. | 2 | 11 | 68.33 | 37.33 | 31.00 | [ckpt](https://huggingface.co/MohammadFazli/xAI-remote-sensing-dior-model/blob/main/best_fixing-wrn_70_16.pth) |
| Standard      | 4 | 99 | 52.33 | 0.00  | 52.33 | [ckpt](https://huggingface.co/MohammadFazli/xAI-remote-sensing-dior-model/blob/main/standard_remote_sensing.pth) |

**Table — Remote Sensing Attribution Coverage.** Percent of attribution mass inside ground-truth boxes at the 5%, 25%, and 50% thresholds for Saliency Maps (SM), DeepLIFT (DL), and Integrated Gradients (IG).

| Model | SM-5 | SM-25 | SM-50 | DL-5 | DL-25 | DL-50 | IG-5 | IG-25 | IG-50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chen et al. [11]   | 7.54 | 31.18 | 55.07 | 11.98 | 38.96 | 61.62 | 11.25 | 39.51 | 62.99 |
| Rade et al. [9]    | 7.86 | 29.12 | 52.07 | 8.40  | 30.31 | 54.07 | 13.30 | 35.86 | 57.80 |
| Rebuffi et al. [7] | 10.51 | 36.32 | 59.87 | 9.89  | 35.28 | 59.46 | 10.50 | 35.50 | 59.50 |
| Standard [31]      | 3.18 | 16.38 | 36.34 | 5.59  | 23.35 | 45.77 | 5.37  | 23.80 | 47.29 |

*Figure — Remote sensing qualitative results.* For the same scene (left, with red bounding boxes), attribution maps from four models are shown across methods (Saliency, DeepLIFT, Integrated Gradients, Occlusion). Robust models (Chen, Rade, Rebuffi) concentrate responses on the target objects, while the standard model yields diffuse, background-biased activations.

<img width="2659" height="1699" alt="Picture2" src="https://github.com/user-attachments/assets/dcd1d2ac-b7d9-40b9-803b-e45406651761" />




