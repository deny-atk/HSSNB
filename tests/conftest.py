import numpy as np
import pytest
from hypothesis import settings
from scipy.io import savemat

# Property tests run the same examples every time, locally and in CI.
settings.register_profile("derandomized", derandomize=True, deadline=None)
settings.load_profile("derandomized")

# Variable names used inside the real .mat files.
MAT_KEYS = {
    "IP": ("indian_pines_corrected", "indian_pines_gt"),
    "PU": ("paviaU", "paviaU_gt"),
    "SA": ("salinas_corrected", "salinas_gt"),
}


@pytest.fixture
def make_scene(tmp_path):
    """Write a small synthetic scene in the same .mat layout as the real datasets.

    Every pixel is labelled, with classes laid out as horizontal stripes, and the
    spectra depend on the class so a model can learn them.
    """

    def make(dataset="IP", bands=200, size=16, num_classes=3, seed=0, background=False):
        rng = np.random.default_rng(seed)
        gt = (np.arange(size)[:, None] * num_classes // size + 1).repeat(size, axis=1)
        if background:  # unlabel the left quarter of the scene
            gt[:, : size // 4] = 0
        signatures = rng.uniform(1000, 5000, size=(num_classes + 1, bands))
        data = signatures[gt] + rng.normal(0, 50, size=(size, size, bands))
        data_key, gt_key = MAT_KEYS[dataset]
        data_path = tmp_path / f"{dataset}_data.mat"
        gt_path = tmp_path / f"{dataset}_gt.mat"
        savemat(data_path, {data_key: data.astype(np.int16)})
        savemat(gt_path, {gt_key: gt.astype(np.uint8)})
        return data_path, gt_path

    return make
