import importlib.util
from pathlib import Path

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import train

_spec = importlib.util.spec_from_file_location(
    "make_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures)


def _brute_force_counts(mask, coords, window):
    half = window // 2
    padded = np.pad(mask, half)
    return np.array([padded[r:r + window, c:c + window].sum() for r, c in coords])


@given(
    mask=arrays(np.int64, st.tuples(st.integers(1, 12), st.integers(1, 12)), elements=st.integers(0, 1)),
    window=st.sampled_from([1, 3, 5, 7]),
)
@example(mask=np.ones((3, 3), dtype=np.int64), window=7)
def test_window_counts_match_brute_force(mask, window):
    coords = np.argwhere(np.ones_like(mask))
    train_coords = np.argwhere(mask)
    counts = make_figures.training_pixels_in_windows(mask.shape, train_coords, coords, window)
    np.testing.assert_array_equal(counts, _brute_force_counts(mask, coords, window))


def test_per_class_accuracy_by_hand():
    acc, support = make_figures.per_class_accuracy(np.array([0, 0, 0, 0, 1, 1]),
                                                   np.array([0, 0, 0, 1, 1, 2]), num_classes=3)
    np.testing.assert_array_equal(support, [4, 2, 0])
    np.testing.assert_allclose(acc[:2], [0.75, 0.5])
    assert np.isnan(acc[2])


def test_figures_render_from_a_trained_checkpoint(make_scene, tmp_path):
    data_path, gt_path = make_scene("IP", size=16, num_classes=3)
    train.main(["--dataset", "IP", "--data_path", str(data_path), "--gt_path", str(gt_path),
                "--num_epochs", "1", "--save_dir", str(tmp_path), "--device", "cpu",
                "--n_components", "13", "--window_size", "11", "--batch_size", "32"])
    pre, split, y_pred = make_figures.predict_test(tmp_path / "best_model_IP.pth", data_path, gt_path,
                                                   "cpu")
    y_true = np.asarray(split["test"].labels)
    assert y_pred.shape == y_true.shape

    acc, support = make_figures.per_class_accuracy(y_true, y_pred, 3)
    results = [("Synthetic", ["a", "b", "c"], acc, support, float(np.mean(y_true == y_pred)))]
    for theme in make_figures.THEMES:
        stats = make_figures.plot_map(pre, split, y_pred, tmp_path / f"map_{theme}.png", theme)
        make_figures.plot_per_class(results, tmp_path / f"per_class_{theme}.png", theme)
        assert (tmp_path / f"map_{theme}.png").stat().st_size > 0
        assert (tmp_path / f"per_class_{theme}.png").stat().st_size > 0
        assert stats["n_test"] == len(y_true)
        assert stats["n_wrong"] == int((y_pred != y_true).sum())
    assert "| a |" in make_figures.markdown_table(results)
