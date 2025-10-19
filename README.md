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

