"""Loading the SVHN cropped-digit .mat files."""
import numpy as np
import scipy.io


def load_svhn(path):
    """Return images (N, 32, 32, 3) uint8 and labels (N,) in 0..9.

    The original files store digit 0 as label 10; this is remapped to 0.
    """
    mat = scipy.io.loadmat(path)
    images = np.transpose(mat["X"], (3, 0, 1, 2))
    labels = mat["y"].ravel().astype(int)
    labels[labels == 10] = 0
    return images, labels


def subsample(images, labels, n, seed=0):
    """Stratified random subset of size n (or everything if n is None)."""
    if n is None or n >= len(labels):
        return images, labels
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=max(1, round(n * np.mean(labels == c))),
                   replace=False)
        for c in np.unique(labels)
    ])
    rng.shuffle(idx)
    return images[idx], labels[idx]
