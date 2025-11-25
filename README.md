#  Unsupervised Anomaly Detection for Supermarket Goods

Detects defective retail items in images using **deep visual features + KNN memory** — no need for labeled defect examples at training!  
A mini-project for Advanced Data Mining (MTech CSE), using the [PKU-GoodsAD](https://github.com/jianzhang96/GoodsAD) dataset and automated feature learning with ResNet18.

---

## 📖 Project Overview

This project offers a robust, easy-to-understand way to **flag defective supermarket products** directly from images.  
Instead of labeling every possible defect, it learns what "normal" products look like, and automatically catches anything that looks visually unusual.

- **Dataset:** PKU-GoodsAD — high-quality, real-world images from 6 product categories, with hundreds of normal and defect cases.
- **Deep Learning Backbone:** Pretrained ResNet18 extracts a 512-dimensional “fingerprint” (feature vector) from each product image.
- **Anomaly Scoring:** K-Nearest Neighbors (KNN, k=1) checks how much a test item deviates from the catalog of normal examples.
- **Evaluation:** Outputs plots, AUROC scores, and side-by-side example images for clear, transparent reporting.

---

## 🚩 Key Features

- **Purely Unsupervised:** Needs only *normal* samples for training.
- **End-to-End Automated:** Data loading, processing, feature extraction, anomaly scoring, and visualization.
- **Cross-Category:** Works identically for drink bottles, cans, food packages, and more.
- **Interpretable Results:** Plots, AUROC, and example images make it easy to explain performance.

---