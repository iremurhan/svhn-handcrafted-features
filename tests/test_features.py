import numpy as np
import scipy.io

from svhn_features.data import load_svhn, subsample
from svhn_features.evaluation import fisher_ratio, fit_and_score, make_models
from svhn_features.features import EXTRACTORS


def _images(n=12, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


def test_every_extractor_returns_fixed_length_finite_vectors():
    imgs = _images()
    for name, fn in EXTRACTORS.items():
        out = fn(imgs)
        assert out.ndim == 2 and out.shape[0] == len(imgs), name
        assert np.isfinite(out).all(), name


def test_hog_dimensions():
    imgs = _images(2)
    assert EXTRACTORS["hog"](imgs).shape[1] == 324        # 3x3 blocks x 2x2 cells x 9 bins
    assert EXTRACTORS["hog_fine"](imgs).shape[1] == 1960  # 7x7 blocks x 2x2 cells x 10 bins


def test_load_svhn_maps_label_10_to_0(tmp_path):
    X = np.zeros((32, 32, 3, 3), dtype=np.uint8)
    path = tmp_path / "toy.mat"
    scipy.io.savemat(path, {"X": X, "y": np.array([[10], [1], [9]])})
    images, labels = load_svhn(path)
    assert images.shape == (3, 32, 32, 3)
    assert labels.tolist() == [0, 1, 9]


def test_fisher_ratio_prefers_separated_classes():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    separated = rng.normal(size=(100, 5)) + 3 * y[:, None]
    mixed = rng.normal(size=(100, 5))
    assert fisher_ratio(separated, y) > fisher_ratio(mixed, y)


def test_models_learn_a_simple_signal():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 60)
    X = rng.normal(size=(180, 20)) + 2 * np.eye(3)[y] @ rng.normal(size=(3, 20))
    tr = rng.permutation(180)[:120]
    te = np.setdiff1d(np.arange(180), tr)
    for name, model in make_models(pca_components=10).items():
        scores = fit_and_score(model, X[tr], y[tr], X[te], y[te])
        assert scores["test_accuracy"] > 0.6, name


def test_subsample_is_stratified():
    labels = np.repeat(np.arange(10), 100)
    _, sub = subsample(np.zeros((1000, 1)), labels, 200)
    assert np.bincount(sub).tolist() == [20] * 10
