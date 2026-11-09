"""Load a hyperspectral scene, reduce it with PCA and serve labelled patches."""

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# 0-based indices of the water-absorption bands that the "corrected" files leave out.
# Source: the "Hyperspectral Remote Sensing Scenes" page of the Grupo de Inteligencia
# Computacional (UPV/EHU), which lists 1-based bands. Indian Pines drops [104-108],
# [150-163] and 220 (220 -> 200 bands); Salinas drops [108-112], [154-167] and 224
# (224 -> 204). Pavia University ships with its noisy bands already removed.
WATER_BANDS = {
    "IP": [*range(103, 108), *range(149, 163), 219],
    "PU": [],
    "SA": [*range(107, 112), *range(153, 167), 223],
}
RAW_BAND_COUNTS = {"IP": 220, "SA": 224}
DATASETS = tuple(WATER_BANDS)


def load_mat_array(path):
    """Return the single array stored in a dataset .mat file."""
    arrays = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    if len(arrays) != 1:
        raise ValueError(f"{path} should hold exactly one array, found {sorted(arrays)}")
    return next(iter(arrays.values()))


def drop_water_bands(data, dataset_name):
    """Drop the water-absorption bands from a raw (uncorrected) scene."""
    bands = WATER_BANDS[dataset_name]
    if not bands:
        return data
    raw = RAW_BAND_COUNTS[dataset_name]
    if data.shape[-1] != raw:
        raise ValueError(
            f"{dataset_name} scene has {data.shape[-1]} bands, but water-band removal needs "
            f"the raw {raw}-band scene. The corrected file has already dropped them."
        )
    return np.delete(data, bands, axis=-1)


def stratified_split(labels, train_size, val_size, seed):
    """Split sample indices class by class into train, validation and test.

    Each class gives int(n * train_size) samples to training and int(n * val_size)
    to validation; the rest go to test.
    """
    if not (0 < train_size and 0 <= val_size and train_size + val_size < 1):
        raise ValueError(
            f"Need 0 < train_size, 0 <= val_size and train_size + val_size < 1; "
            f"got {train_size} and {val_size}"
        )
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train, n_val = int(len(idx) * train_size), int(len(idx) * val_size)
        parts[0].append(idx[:n_train])
        parts[1].append(idx[n_train:n_train + n_val])
        parts[2].append(idx[n_train + n_val:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


class PatchDataset(Dataset):
    """Labelled pixels of one scene; each item is the window centred on a pixel.

    Patches are cut on access from the padded cube, so memory stays at the size of
    the cube instead of one copy of the window per pixel.
    """

    def __init__(self, cube, coords, labels, window_size):
        self.cube = cube  # (n_components, H + window_size - 1, W + window_size - 1)
        self.coords = coords  # (N, 2) row and column in the unpadded scene
        self.labels = labels  # (N,) 0-based class indices
        self.window_size = window_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row, col = self.coords[idx].tolist()
        patch = self.cube[:, row:row + self.window_size, col:col + self.window_size]
        return patch.unsqueeze(0), self.labels[idx]  # (1, n_components, window, window)


class HSIPreprocessor:
    """Preprocessor for the Indian Pines, Pavia University and Salinas scenes."""

    def __init__(self, data_path, gt_path, dataset_name, n_components=30,
                 window_size=25, remove_water_bands=False):
        self.dataset_name = dataset_name.upper()
        if self.dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Use one of {DATASETS}")
        self.n_components = n_components
        self.window_size = window_size

        self.data = load_mat_array(Path(data_path))
        self.gt = load_mat_array(Path(gt_path))
        if self.data.shape[:2] != self.gt.shape:
            raise ValueError(f"Scene {self.data.shape[:2]} and ground truth {self.gt.shape} differ")
        if remove_water_bands:
            self.data = drop_water_bands(self.data, self.dataset_name)
        self.data_pca = None

    def apply_pca(self):
        """Standardise every band, then keep the first n_components principal components.

        PCA is fitted on all pixels of the scene, labelled or not, as in the paper.
        The exact solver keeps the result identical between train.py and evaluate.py;
        scikit-learn < 1.5 would otherwise pick an unseeded randomized solver here.
        """
        h, w, c = self.data.shape
        scaled = StandardScaler().fit_transform(self.data.reshape(-1, c).astype(np.float64))
        reduced = PCA(n_components=self.n_components, svd_solver="full").fit_transform(scaled)
        self.data_pca = reduced.reshape(h, w, self.n_components).astype(np.float32)
        return self.data_pca

    def preprocess_pipeline(self, train_size=0.3, val_size=0.1, seed=42):
        """Run PCA and return train, validation and test PatchDatasets."""
        self.apply_pca()
        pad = self.window_size // 2
        padded = np.pad(self.data_pca, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        cube = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))

        rows, cols = np.nonzero(self.gt)  # label 0 is unlabelled background
        coords = torch.from_numpy(np.stack([rows, cols], axis=1)).long()
        labels = torch.from_numpy(self.gt[rows, cols].astype(np.int64) - 1)

        parts = stratified_split(labels.numpy(), train_size, val_size, seed)
        split = {
            name: PatchDataset(cube, coords[idx], labels[idx], self.window_size)
            for name, idx in zip(("train", "val", "test"), parts)
        }
        split["num_classes"] = int(self.gt.max())
        return split
