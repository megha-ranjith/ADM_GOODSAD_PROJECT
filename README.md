#  Unsupervised Anomaly Detection for Supermarket Goods

Detects defective retail items in images using **deep visual features + KNN memory** — no need for labeled defect examples at training!  
A micro-project for Advanced Data Mining (MTech CSE), using the [PKU-GoodsAD](https://github.com/jianzhang96/GoodsAD) dataset and automated feature learning with ResNet18.

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

## 📦 Repository Structure
```
ADM_GOODSAD_PROJECT/
│
├── main.py # Main end-to-end project code
├── images/ # Plots: score histograms, good/defect sample pairs
├── GoodsAD/ # PKU-GoodsAD dataset (not included)
├── requirements.txt # Python dependencies
└── README.md
```

---
## 📝 Usage

### 1. Clone the repo
```bash
git clone https://github.com/megha-ranjith/ADM_GOODSAD_PROJECT.git
cd ADM_GOODSAD_PROJECT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download/organize the PKU-GoodsAD dataset
- Place your data under `GoodsAD/` directory, following the [official structure](https://github.com/jianzhang96/GoodsAD).

### 4. Run the main script
```bash
python main.py
```
- All key outputs (plots & sample images) are saved in the `images/` folder.

---

## 💡 Methodology

- **Only uses normal (“good”) examples for memory.**
- Each image is processed with ResNet18 (final layer removed) to get a 512-D feature vector.
- Build a KNN (k=1) “memory bank” of all normal features per category.
- At test time, each image is scored by its distance to the closest normal — big distances signal a likely defect.
- Plots and metrics generated for each category.

---

## 📈 Outputs & Visualizations

- `images/<category>_hist.png`: Histogram showing anomaly scores for good (green) vs. defect (red) images for each product category.
- `images/<category>_good_defect_pair.png`: Side-by-side example — normal sample (left) and defect (right).

---

## 👩‍💻 Author

- Megha Ranjith

---

## 📝 License

MIT License

---
