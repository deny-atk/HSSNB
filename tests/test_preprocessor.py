import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.io import savemat

from hssnb.preprocessor import HSIPreprocessor, drop_water_bands, load_mat_array, stratified_split

IP_REMOVED_1BASED = [*range(104, 109), *range(150, 164), 220]
SA_REMOVED_1BASED = [*range(108, 113), *range(154, 168), 224]


@pytest.mark.parametrize(("dataset", "bands"), [("IP", 200), ("PU", 103), ("SA", 204)])
def test_dw1_pipeline_returns_patch_datasets(make_scene, dataset, bands):
    data_path, gt_path = make_scene(dataset, bands=bands, size=16, num_classes=3)
    pre = HSIPreprocessor(data_path, gt_path, dataset, n_components=15, window_size=13)
    split = pre.preprocess_pipeline(train_size=0.3, val_size=0.1, seed=0)

    assert split["num_classes"] == 3
    assert sum(len(split[name]) for name in ("train", "val", "test")) == 16 * 16
    patch, label = split["train"][0]
    assert patch.shape == (1, 15, 13, 13)
    assert 0 <= int(label) < 3
    for name in ("train", "val", "test"):
        assert all(torch.isfinite(p).all() for p, _ in split[name])


def test_dw1_background_pixels_are_not_samples(make_scene):
    data_path, gt_path = make_scene("IP", size=16, background=True)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=15, window_size=13)
    split = pre.preprocess_pipeline(seed=0)

    used = sorted(tuple(c) for name in ("train", "val", "test") for c in split[name].coords.tolist())
    assert used == sorted(zip(*np.nonzero(pre.gt)))


def test_dw1_patches_match_a_reflected_window_reference(make_scene):
    data_path, gt_path = make_scene("IP", size=16)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=15, window_size=13)
    split = pre.preprocess_pipeline(seed=0)
    test = split["test"]
    reference = np.pad(pre.data_pca, ((6, 6), (6, 6), (0, 0)), mode="reflect")
    # Corners and edges exercise the reflected border; the middle needs no padding.
    wanted = {(0, 0), (0, 15), (15, 0), (15, 15), (7, 0), (0, 7), (8, 8)}
    seen = set()
    for k in range(len(test)):
        row, col = test.coords[k].tolist()
        if (row, col) not in wanted:
            continue
        seen.add((row, col))
        patch, label = test[k]
        window = reference[row:row + 13, col:col + 13].transpose(2, 0, 1)
        np.testing.assert_array_equal(patch[0].numpy(), window)
        np.testing.assert_array_equal(patch[0, :, 6, 6].numpy(), pre.data_pca[row, col])
        assert int(label) == pre.gt[row, col] - 1
    assert seen, "no corner, edge or centre pixel landed in the test split"


def test_dw1_pca_matches_an_svd_reference(make_scene):
    data_path, gt_path = make_scene("IP", size=16)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=5)
    reduced = pre.apply_pca().reshape(-1, 5)

    x = pre.data.reshape(-1, pre.data.shape[-1]).astype(np.float64)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    expected = u[:, :5] * s[:5]
    # Component signs are arbitrary; compare each column up to sign.
    for k in range(5):
        sign = np.sign(expected[:, k] @ reduced[:, k])
        np.testing.assert_allclose(reduced[:, k], sign * expected[:, k], rtol=1e-4, atol=1e-3)


def test_dw1_mat_file_must_hold_one_array(tmp_path):
    path = tmp_path / "two.mat"
    savemat(path, {"a": np.zeros(2), "b": np.zeros(2)})
    with pytest.raises(ValueError, match="exactly one array"):
        load_mat_array(path)


def test_dw1_scene_and_ground_truth_sizes_must_match(tmp_path):
    savemat(tmp_path / "data.mat", {"x": np.zeros((4, 4, 200))})
    savemat(tmp_path / "gt.mat", {"y": np.zeros((4, 5))})
    with pytest.raises(ValueError, match="differ"):
        HSIPreprocessor(tmp_path / "data.mat", tmp_path / "gt.mat", "IP")


@pytest.mark.parametrize(
    ("dataset", "raw_bands", "removed_1based"),
    [("IP", 220, IP_REMOVED_1BASED), ("SA", 224, SA_REMOVED_1BASED)],
)
def test_dw2_water_bands_removed_from_raw_scene(dataset, raw_bands, removed_1based):
    # Band b holds the value b, so the kept values name the kept bands, in order.
    data = np.broadcast_to(np.arange(1, raw_bands + 1), (2, 2, raw_bands))
    kept = drop_water_bands(data, dataset)
    expected = [b for b in range(1, raw_bands + 1) if b not in removed_1based]
    assert len(expected) == raw_bands - 20
    assert kept[0, 0].tolist() == expected


@given(cube=arrays(np.float32, (2, 3, 220), elements=st.floats(-1e3, 1e3, width=32)))
def test_dw2_removal_keeps_every_other_value(cube):
    keep = [b - 1 for b in range(1, 221) if b not in IP_REMOVED_1BASED]
    np.testing.assert_array_equal(drop_water_bands(cube, "IP"), cube[:, :, keep])


@pytest.mark.parametrize(("dataset", "bands"), [("IP", 200), ("SA", 204)])
def test_dw2_corrected_scene_with_removal_raises(dataset, bands):
    with pytest.raises(ValueError, match="already"):
        drop_water_bands(np.zeros((2, 2, bands)), dataset)


def test_dw2_pavia_has_no_water_bands():
    data = np.random.default_rng(0).normal(size=(2, 2, 103))
    np.testing.assert_array_equal(drop_water_bands(data, "PU"), data)


def test_dw2_preprocessor_removes_bands_when_asked(make_scene):
    data_path, gt_path = make_scene("IP", bands=220)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=15, remove_water_bands=True)
    assert pre.data.shape[-1] == 200


@given(
    labels=st.lists(st.integers(0, 4), min_size=1, max_size=200),
    train_size=st.floats(0.05, 0.6),
    val_size=st.floats(0.0, 0.3),
    seed=st.integers(0, 2**32 - 1),
)
@example(labels=[0], train_size=0.3, val_size=0.1, seed=0)
def test_dw4_split_partitions_each_class(labels, train_size, val_size, seed):
    labels = np.array(labels)
    train, val, test = stratified_split(labels, train_size, val_size, seed)

    everything = np.concatenate([train, val, test])
    assert len(everything) == len(labels)
    assert set(everything.tolist()) == set(range(len(labels)))
    for c in np.unique(labels):
        n = int((labels == c).sum())
        assert int((labels[train] == c).sum()) == int(n * train_size)
        assert int((labels[val] == c).sum()) == int(n * val_size)


@pytest.mark.parametrize(("train_size", "val_size"), [(0.0, 0.1), (0.7, 0.3), (0.3, -0.1)])
def test_dw4_split_sizes_validated(train_size, val_size):
    with pytest.raises(ValueError):
        stratified_split(np.zeros(10, dtype=int), train_size, val_size, seed=0)


def test_dw5_same_seed_same_split():
    labels = np.repeat(np.arange(4), 25)
    first = stratified_split(labels, 0.3, 0.1, seed=7)
    again = stratified_split(labels, 0.3, 0.1, seed=7)
    other = stratified_split(labels, 0.3, 0.1, seed=8)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], other[0])


def test_unknown_dataset_rejected(make_scene):
    data_path, gt_path = make_scene("IP")
    with pytest.raises(ValueError, match="Unsupported dataset"):
        HSIPreprocessor(data_path, gt_path, "HO")
