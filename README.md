# Hand-Crafted Features for SVHN Digit Classification

How far do classical image descriptors and non-deep classifiers get on the Street View House Numbers (SVHN) digits? This repository extracts eight feature sets (HOG at two resolutions, LBP, GLCM, Gabor, Sobel, colour histograms, raw greyscale pixels), measures how class-separable each one is, and compares k-NN, LDA, logistic regression, decision trees and random forests, with and without PCA.

Based on a two-person term project for CMPE 544 (Pattern Recognition), Boğaziçi University, Fall 2023. This repository covers my part, feature extraction and the classical classifiers, refactored in 2026.

## Data

Not included. Download the cropped-digit files `train_32x32.mat` and `test_32x32.mat` from the [SVHN website](http://ufldl.stanford.edu/housenumbers/) (non-commercial use only) into `data/`. Label 10 in the files is mapped to digit 0.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full comparison (all features x all models) on the official splits
python main.py --train data/train_32x32.mat --test data/test_32x32.mat --out results.csv

# quicker: a stratified 20k training subset and selected features/models
python main.py --train data/train_32x32.mat --test data/test_32x32.mat \
    --train-size 20000 --features hog hog_fine gray_pixels --models knn random_forest

pytest -q
```

`results.csv` has one row per (feature, model) with train/test accuracy, feature dimension, Fisher ratio and runtime.

## Method notes

- **Distinctiveness** is measured with the Fisher trace ratio (between-class over within-class scatter of standardised features).
- **Scaling and PCA (100 components)** are fitted inside each scikit-learn pipeline on training data only, so test images are transformed exactly like training images.
- `hog` uses 8×8-pixel cells and 9 orientations (324-D); `hog_fine` uses 4×4-pixel cells and 10 orientations (1,960-D).

## Results

Results on a reduced split (8,000 train / 4,000 held-out images, both drawn from the official *test* file):

| Feature | k-NN | Random forest | Random forest + PCA |
|---|---|---|---|
| HOG, fine (1,960-D) | 0.794 | 0.791 | 0.778 |
| HOG (324-D) | 0.784 | 0.764 | 0.742 |
| Greyscale pixels | 0.478 | 0.635 | 0.488 |
| LBP histogram (10-D) | 0.189 | 0.204 | 0.202 |

These are indicative only and not comparable with published SVHN numbers.
