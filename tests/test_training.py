import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import evaluate as evaluate_script
import train
from hssnb.training import compute_metrics, resolve_device


@pytest.mark.parametrize(
    ("name", "cuda", "mps", "expected"),
    [
        ("mps", False, True, "mps"),
        ("mps", False, False, "cpu"),
        ("cuda", True, False, "cuda"),
        ("cuda", False, True, "cpu"),
        ("cuda:1", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
    ],
    ids=["mps-present", "mps-absent", "cuda-present", "cuda-absent", "cuda-index-absent", "cpu"],
)
def test_dw7_requested_device_or_cpu_fallback(monkeypatch, name, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device(name).type == expected


@pytest.mark.parametrize("script", [train, evaluate_script], ids=["train", "evaluate"])
def test_dw7_scripts_pass_the_device_flag_through(monkeypatch, script):
    requested = []

    def spy(name):
        requested.append(name)
        raise SystemExit  # stop before any data is read

    monkeypatch.setattr(script, "resolve_device", spy)
    with pytest.raises(SystemExit):
        script.main(["--dataset", "IP", "--data_path", "x.mat", "--gt_path", "y.mat", "--device", "mps"])
    assert requested == ["mps"]


def test_metrics_by_hand():
    # Class 0: 3 of 4 right; class 1: 1 of 2 right; class 2: never in y_true.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 2])
    m = compute_metrics(y_true, y_pred, num_classes=3)

    np.testing.assert_array_equal(m["confusion_matrix"], [[3, 1, 0], [0, 1, 1], [0, 0, 0]])
    assert m["OA"] == 4 / 6
    assert m["AA"] == (3 / 4 + 1 / 2) / 2  # class 2 is absent, so it is not averaged in
    # Cohen's kappa: observed 4/6, chance (4*3 + 2*2 + 0*1) / 36 = 16/36.
    np.testing.assert_allclose(m["Kappa"], (4 / 6 - 16 / 36) / (1 - 16 / 36))


@given(
    pairs=st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3)), min_size=1, max_size=60),
    extra_classes=st.integers(0, 3),
)
def test_metrics_match_a_counting_reference(pairs, extra_classes):
    y_true = np.array([t for t, _ in pairs])
    y_pred = np.array([p for _, p in pairs])
    m = compute_metrics(y_true, y_pred, num_classes=4 + extra_classes)

    assert m["OA"] == sum(t == p for t, p in pairs) / len(pairs)
    per_class = [
        sum(t == p == c for t, p in pairs) / sum(t == c for t, _ in pairs)
        for c in sorted(set(y_true.tolist()))
    ]
    np.testing.assert_allclose(m["AA"], np.mean(per_class))
    assert m["confusion_matrix"].sum() == len(pairs)
