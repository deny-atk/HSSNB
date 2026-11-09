import re

import numpy as np
import pytest
import torch
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.io import loadmat, savemat

import evaluate as evaluate_script
import train
from hssnb.preprocessor import HSIPreprocessor, spatial_split

FAST = ["--n_components", "13", "--window_size", "11", "--batch_size", "64", "--device", "cpu"]


def chebyshev(a, b):
    """Pairwise Chebyshev distances between two (N, 2) coordinate arrays."""
    return np.abs(a[:, None, :] - b[None, :, :]).max(axis=-1)


def assert_guarded(coords, train, val, test, window_size):
    pad = window_size // 2
    seen = coords[np.concatenate([train, val])]
    if len(test) and len(seen):
        assert chebyshev(coords[test], seen).min() > pad
    if len(val) and len(train):
        assert chebyshev(coords[val], coords[train]).min() > pad


def islet_scene(make_scene, size=32):
    """A make_scene scene plus a fourth class confined to a 2x2 patch in one corner."""
    data_path, gt_path = make_scene("IP", size=size, num_classes=3)
    gt = loadmat(gt_path)["indian_pines_gt"]
    gt[:2, :2] = 4
    savemat(gt_path, {"indian_pines_gt": gt})
    return data_path, gt_path


@given(
    gt=st.tuples(st.integers(1, 24), st.integers(1, 24)).flatmap(
        lambda shape: arrays(np.uint8, shape, elements=st.integers(0, 4))),
    window_size=st.integers(1, 11),
    block_size=st.integers(1, 12),
    train_size=st.floats(0.05, 0.8),
    val_size=st.floats(0.0, 0.3),
    seed=st.integers(0, 2**32 - 1),
)
@example(gt=np.ones((1, 1), np.uint8), window_size=25, block_size=1, train_size=0.3, val_size=0.1, seed=0)
@example(gt=np.zeros((3, 3), np.uint8), window_size=3, block_size=1, train_size=0.3, val_size=0.1, seed=0)
def test_dw1_spatial_split_keeps_test_out_of_every_training_window(
        gt, window_size, block_size, train_size, val_size, seed):
    assume(train_size + val_size < 1)
    train_idx, val, test = spatial_split(gt, window_size, train_size, val_size, seed, block_size)
    rows, cols = np.nonzero(gt)
    coords = np.stack([rows, cols], axis=1)

    parts = np.concatenate([train_idx, val, test])
    assert len(set(parts.tolist())) == len(parts)  # no pixel in two splits
    assert set(parts.tolist()) <= set(range(len(rows)))
    for part in (train_idx, val, test):
        np.testing.assert_array_equal(part, np.sort(part))
    assert_guarded(coords, train_idx, val, test, window_size)
    # dw2: every labelled class has training pixels, whatever the layout.
    assert set(gt[rows[train_idx], cols[train_idx]].tolist()) == set(gt[rows, cols].tolist())


def test_dw1_pipeline_spatial_split_is_guarded(make_scene):
    data_path, gt_path = make_scene("IP", size=40, num_classes=3)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=7)
    split = pre.preprocess_pipeline(0.3, 0.1, seed=0, split="spatial", block_size=8)

    coords = np.concatenate([split[n].coords.numpy() for n in ("train", "val", "test")])
    sizes = np.cumsum([len(split[n]) for n in ("train", "val", "test")])
    train_idx, val, test = np.split(np.arange(sizes[-1]), sizes[:-1])
    assert all(len(p) for p in (train_idx, val, test))
    assert_guarded(coords, train_idx, val, test, 7)
    # Labels still match the ground truth at each patch centre.
    for name in ("train", "val", "test"):
        rows, cols = split[name].coords.numpy().T
        np.testing.assert_array_equal(split[name].labels.numpy(), pre.gt[rows, cols] - 1)


def test_dw1_spatial_split_is_much_further_apart_than_random(make_scene):
    data_path, gt_path = make_scene("IP", size=40, num_classes=3)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=7)
    nearest = {}
    for mode in ("random", "spatial"):
        split = pre.preprocess_pipeline(0.3, 0.1, seed=0, split=mode, block_size=8)
        nearest[mode] = chebyshev(split["test"].coords.numpy(), split["train"].coords.numpy()).min()
    assert nearest["random"] <= 1  # random sampling puts training pixels right next to test pixels
    assert nearest["spatial"] > 7 // 2


def test_dw1_unknown_split_rejected(make_scene):
    data_path, gt_path = make_scene("IP")
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=7)
    with pytest.raises(ValueError, match="split"):
        pre.preprocess_pipeline(split="checkerboard")


@pytest.mark.parametrize(("train_size", "val_size"), [(0.0, 0.1), (0.7, 0.3), (0.3, -0.1)])
def test_dw1_spatial_split_sizes_validated(train_size, val_size):
    with pytest.raises(ValueError):
        spatial_split(np.ones((10, 10), np.uint8), 3, train_size, val_size, seed=0)


def test_dw2_roomy_scene_has_every_class_in_every_split(make_scene):
    data_path, gt_path = make_scene("IP", size=60, num_classes=3)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=5)
    split = pre.preprocess_pipeline(0.3, 0.1, seed=0, split="spatial", block_size=10)
    for name in ("train", "val", "test"):
        assert sorted(set(split[name].labels.tolist())) == [0, 1, 2]
    assert split["missing_classes"] == {"train": [], "val": [], "test": []}


def test_dw2_class_in_one_block_goes_to_train_and_is_reported_missing_elsewhere(make_scene):
    data_path, gt_path = islet_scene(make_scene)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=7)
    for seed in range(5):
        split = pre.preprocess_pipeline(0.3, 0.1, seed=seed, split="spatial", block_size=8)
        assert 3 in split["train"].labels.tolist()  # 0-based index of class 4
        assert split["missing_classes"]["train"] == []
        assert 4 in split["missing_classes"]["val"]
        assert 4 in split["missing_classes"]["test"]


def test_dw2_random_split_also_reports_missing_classes(make_scene):
    # Four pixels of class 4: int(4 * 0.1) == 0 of them go to validation.
    data_path, gt_path = islet_scene(make_scene, size=16)
    pre = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=7)
    split = pre.preprocess_pipeline(0.3, 0.1, seed=0)
    assert split["missing_classes"] == {"train": [], "val": [4], "test": []}


def run_train(data_path, gt_path, save_dir, extra=()):
    train.main([
        "--dataset", "IP", "--data_path", str(data_path), "--gt_path", str(gt_path),
        "--num_epochs", "2", "--seed", "0", "--save_dir", str(save_dir), *FAST, *extra,
    ])
    checkpoint = torch.load(save_dir / "best_model_IP.pth", weights_only=True)
    return checkpoint, np.load(save_dir / "metrics_IP.npz")


def run_evaluate(data_path, gt_path, save_dir):
    return evaluate_script.main([
        "--dataset", "IP", "--checkpoints_dir", str(save_dir),
        "--data_path", str(data_path), "--gt_path", str(gt_path), "--device", "cpu",
    ])


def test_dw2_train_reports_missing_classes_and_finishes(make_scene, tmp_path, capsys):
    data_path, gt_path = islet_scene(make_scene)
    run_train(data_path, gt_path, tmp_path, extra=["--split", "spatial", "--block_size", "8"])
    out = capsys.readouterr().out
    # Class 4 is the highest class, so it ends each list; the guard can drop others too.
    assert re.search(r"^Classes with no validation samples: \[.*\b4\]$", out, re.M)
    assert re.search(r"^Classes with no test samples: \[.*\b4\]$", out, re.M)
    assert "Classes with no training samples" not in out
    assert (tmp_path / "metrics_IP.npz").exists()


def test_dw2_empty_test_split_rejected_before_training(make_scene, tmp_path):
    # One block holds the whole scene, so it all goes to training.
    data_path, gt_path = make_scene("IP", size=16, num_classes=3)
    with pytest.raises(ValueError, match="Empty split"):
        run_train(data_path, gt_path, tmp_path, extra=["--split", "spatial", "--block_size", "16"])
    assert not (tmp_path / "best_model_IP.pth").exists()


def test_dw3_evaluate_reproduces_spatial_split_test_metrics(make_scene, tmp_path):
    data_path, gt_path = make_scene("IP", size=32, num_classes=3)
    checkpoint, metrics = run_train(data_path, gt_path, tmp_path,
                                    extra=["--split", "spatial", "--block_size", "8"])
    assert checkpoint["config"]["split"] == "spatial"
    assert checkpoint["config"]["block_size"] == 8

    result = run_evaluate(data_path, gt_path, tmp_path)

    assert result["OA"] == float(metrics["test_oa"])
    np.testing.assert_array_equal(result["confusion_matrix"], metrics["test_confusion_matrix"])
    # The spatial test split is smaller than the random one, so a fallback to the
    # random split could not give the same confusion matrix.
    random_test = HSIPreprocessor(data_path, gt_path, "IP", n_components=13, window_size=11
                                  ).preprocess_pipeline(0.3, 0.1, 0)["test"]
    assert result["confusion_matrix"].sum() < len(random_test)


def test_dw3_same_seed_same_spatial_split():
    gt = np.random.default_rng(0).integers(0, 4, size=(40, 40)).astype(np.uint8)
    first = spatial_split(gt, 5, 0.3, 0.1, seed=7, block_size=8)
    again = spatial_split(gt, 5, 0.3, 0.1, seed=7, block_size=8)
    other = spatial_split(gt, 5, 0.3, 0.1, seed=8, block_size=8)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_dw3_checkpoint_from_before_the_option_still_uses_the_random_split(make_scene, tmp_path):
    data_path, gt_path = make_scene("IP", size=16, num_classes=3)
    checkpoint, metrics = run_train(data_path, gt_path, tmp_path)
    assert checkpoint["config"]["split"] == "random"
    for key in ("split", "block_size"):
        del checkpoint["config"][key]
    torch.save(checkpoint, tmp_path / "best_model_IP.pth")

    result = run_evaluate(data_path, gt_path, tmp_path)

    assert result["OA"] == float(metrics["test_oa"])
    np.testing.assert_array_equal(result["confusion_matrix"], metrics["test_confusion_matrix"])
