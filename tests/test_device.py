import pytest
import torch

import evaluate as evaluate_script
import train


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
    from hssnb.training import resolve_device

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
