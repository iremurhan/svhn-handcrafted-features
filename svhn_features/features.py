"""Hand-crafted image descriptors for 32x32 RGB digits.

Every extractor maps an (N, 32, 32, 3) uint8 array to an (N, D) float array.
"""
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern


def _gray(images):
    return np.stack([rgb2gray(img) for img in images])  # floats in [0, 1]


def hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    return np.stack([
        hog(g, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, block_norm="L2-Hys")
        for g in _gray(images)
    ])


def hog_fine_features(images):
    """'Custom HOG' from the project: 10 orientations on 4x4-pixel cells."""
    return hog_features(images, orientations=10, pixels_per_cell=(4, 4))


def lbp_features(images, points=8, radius=1):
    """Histogram of uniform local binary patterns (points + 2 bins)."""
    bins = np.arange(points + 3)
    out = []
    for g in _gray(images):
        codes = local_binary_pattern((g * 255).astype(np.uint8), points, radius, method="uniform")
        hist, _ = np.histogram(codes, bins=bins)
        out.append(hist / max(hist.sum(), 1))
    return np.stack(out)


def glcm_features(images, distances=(1, 2), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    """Grey-level co-occurrence statistics (contrast, homogeneity, energy, correlation)."""
    props = ("contrast", "homogeneity", "energy", "correlation")
    out = []
    for g in _gray(images):
        glcm = graycomatrix((g * 255).astype(np.uint8), distances, angles, symmetric=True,
                            normed=True)
        out.append(np.concatenate([graycoprops(glcm, p).ravel() for p in props]))
    return np.nan_to_num(np.stack(out))


def gabor_features(images, thetas=(0, 45, 90, 135), ksize=7):
    """Mean and standard deviation of Gabor responses on a 4x4 grid, per orientation."""
    kernels = [cv2.getGaborKernel((ksize, ksize), 4.0, np.deg2rad(t), 10.0, 0.5, 0,
                                  ktype=cv2.CV_32F) for t in thetas]
    out = []
    for g in _gray(images).astype(np.float32):
        feats = []
        for k in kernels:
            resp = cv2.filter2D(g, -1, k).reshape(4, 8, 4, 8)
            feats.extend([resp.mean(axis=(1, 3)).ravel(), resp.std(axis=(1, 3)).ravel()])
        out.append(np.concatenate(feats))
    return np.stack(out)


def sobel_features(images, cells=4):
    """Gradient-magnitude energy pooled on a cells x cells grid."""
    out = []
    for g in _gray(images).astype(np.float32):
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        s = 32 // cells
        out.append(mag.reshape(cells, s, cells, s).mean(axis=(1, 3)).ravel())
    return np.stack(out)


def color_histogram_features(images, bins=16):
    """Per-channel normalised intensity histograms."""
    out = []
    for img in images:
        hists = [np.histogram(img[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
        h = np.concatenate(hists).astype(float)
        out.append(h / max(h.sum(), 1))
    return np.stack(out)


def gray_pixels(images):
    """Flattened greyscale image (1,024 values), the raw-pixel baseline."""
    return _gray(images).reshape(len(images), -1)


EXTRACTORS = {
    "hog": hog_features,
    "hog_fine": hog_fine_features,
    "lbp": lbp_features,
    "glcm": glcm_features,
    "gabor": gabor_features,
    "sobel": sobel_features,
    "color_hist": color_histogram_features,
    "gray_pixels": gray_pixels,
}
