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
    assert resolve_device(name) == torch.device(expected)


# cpu:0 stands in for an accelerator: it differs from the plain "cpu" the scripts used
# to fall back to, and it runs anywhere.
MARKED = torch.device("cpu", 0)
FAST = ["--n_components", "13", "--window_size", "11", "--batch_size", "32"]


def _record_device_then_stop(seen, position):
    def spy(*args):
        seen.append(args[position])  # the device argument
        raise SystemExit
    return spy


def _resolver(requested):
    def resolve(name):
        requested.append(name)
        return MARKED
    return resolve


def test_dw7_train_uses_the_resolved_device(monkeypatch, make_scene, tmp_path):
    data_path, gt_path = make_scene("IP", size=16, num_classes=3)
    requested, seen = [], []
    monkeypatch.setattr(train, "resolve_device", _resolver(requested))
    monkeypatch.setattr(train, "train_one_epoch", _record_device_then_stop(seen, 4))
    with pytest.raises(SystemExit):
        train.main(["--dataset", "IP", "--data_path", str(data_path), "--gt_path", str(gt_path),
                    "--num_epochs", "1", "--save_dir", str(tmp_path), "--device", "mps", *FAST])
    assert requested == ["mps"]
    assert seen == [MARKED]


def test_dw7_evaluate_uses_the_resolved_device(monkeypatch, make_scene, tmp_path):
    data_path, gt_path = make_scene("IP", size=16, num_classes=3)
    train.main(["--dataset", "IP", "--data_path", str(data_path), "--gt_path", str(gt_path),
                "--num_epochs", "1", "--save_dir", str(tmp_path), "--device", "cpu", *FAST])
    requested, seen = [], []
    monkeypatch.setattr(evaluate_script, "resolve_device", _resolver(requested))
    monkeypatch.setattr(evaluate_script, "evaluate", _record_device_then_stop(seen, 3))
    with pytest.raises(SystemExit):
        evaluate_script.main(["--dataset", "IP", "--checkpoints_dir", str(tmp_path),
                              "--data_path", str(data_path), "--gt_path", str(gt_path), "--device", "mps"])
    assert requested == ["mps"]
    assert seen == [MARKED]
