import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import evaluate as evaluate_script
import train
from hssnb.models import HSSNB
from hssnb.preprocessor import HSIPreprocessor
from hssnb.training import evaluate

# Small settings so a run takes about a second on CPU.
FAST = ["--n_components", "13", "--window_size", "11", "--batch_size", "32", "--device", "cpu"]


def run_train(make_scene, save_dir, dataset="IP", bands=200, epochs=3, seed=0, extra=()):
    data_path, gt_path = make_scene(dataset, bands=bands, size=16, num_classes=3)
    train.main([
        "--dataset", dataset, "--data_path", str(data_path), "--gt_path", str(gt_path),
        "--num_epochs", str(epochs), "--seed", str(seed), "--save_dir", str(save_dir), *FAST, *extra,
    ])
    checkpoint = torch.load(save_dir / f"best_model_{dataset}.pth", weights_only=True)
    metrics = np.load(save_dir / f"metrics_{dataset}.npz")
    return data_path, gt_path, checkpoint, metrics


def test_dw1_train_runs_end_to_end(make_scene, tmp_path):
    for dataset, bands in (("IP", 200), ("PU", 103), ("SA", 204)):
        _, _, checkpoint, metrics = run_train(make_scene, tmp_path / dataset, dataset, bands, epochs=3)
        assert checkpoint["config"]["dataset"] == dataset
        assert checkpoint["config"]["num_classes"] == 3
        # The synthetic classes are easy to separate, so a working loop learns them.
        assert metrics["train_losses"][-1] < metrics["train_losses"][0]
        assert float(metrics["test_oa"]) > 0.6  # chance is 1/3


def test_dw4_checkpoint_chosen_on_validation_and_test_scored_once(make_scene, tmp_path):
    data_path, gt_path, checkpoint, metrics = run_train(make_scene, tmp_path, epochs=3)
    config = checkpoint["config"]

    assert len(metrics["val_oa"]) == 3
    assert checkpoint["best_epoch"] == int(np.argmax(metrics["val_oa"])) + 1
    assert np.ndim(metrics["test_oa"]) == 0  # one test score, not one per epoch

    # Re-score the saved weights: val_oa must come from the validation split, which
    # shares no pixel with the test split.
    split = HSIPreprocessor(
        data_path, gt_path, "IP", n_components=config["n_components"],
        window_size=config["window_size"],
    ).preprocess_pipeline(config["train_size"], config["val_size"], config["seed"])
    val_pixels = {tuple(c) for c in split["val"].coords.tolist()}
    test_pixels = {tuple(c) for c in split["test"].coords.tolist()}
    assert val_pixels and not val_pixels & test_pixels

    model = HSSNB(num_classes=3, n_components=config["n_components"])
    model.load_state_dict(checkpoint["model_state"])
    _, val = evaluate(model, nn.CrossEntropyLoss(), DataLoader(split["val"], batch_size=64),
                      "cpu", 3)
    assert val["OA"] == metrics["val_oa"][checkpoint["best_epoch"] - 1]


def test_dw5_same_seed_same_weights_and_metrics(make_scene, tmp_path):
    _, _, first, first_metrics = run_train(make_scene, tmp_path / "a", epochs=2, seed=3)
    _, _, again, again_metrics = run_train(make_scene, tmp_path / "b", epochs=2, seed=3)

    for name, tensor in first["model_state"].items():
        torch.testing.assert_close(tensor, again["model_state"][name], rtol=0, atol=0)
    for key in first_metrics.files:
        np.testing.assert_array_equal(first_metrics[key], again_metrics[key])


def test_dw6_evaluate_uses_checkpoint_settings(make_scene, tmp_path):
    # 13 components and 11x11 windows: evaluate.py's old fixed defaults (30, 25, 16 classes)
    # would build the wrong model and the wrong patches here.
    data_path, gt_path, _, metrics = run_train(make_scene, tmp_path, epochs=2)

    result = evaluate_script.main([
        "--dataset", "IP", "--checkpoints_dir", str(tmp_path),
        "--data_path", str(data_path), "--gt_path", str(gt_path), "--device", "cpu",
    ])

    assert result["confusion_matrix"].shape == (3, 3)
    assert result["OA"] == float(metrics["test_oa"])
    assert (tmp_path / "eval_results" / "IP" / "confusion_matrix.png").exists()


def test_dw4_validation_each_epoch_then_test_once_with_best_weights(make_scene, tmp_path, monkeypatch):
    calls = []

    def recording_evaluate(model, criterion, loader, device, num_classes):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        calls.append((loader.dataset, state))
        return evaluate(model, criterion, loader, device, num_classes)

    monkeypatch.setattr(train, "evaluate", recording_evaluate)
    _, _, checkpoint, _ = run_train(make_scene, tmp_path, epochs=3)

    val_set, test_set = calls[0][0], calls[-1][0]
    assert len(calls) == 3 + 1
    assert all(dataset is val_set for dataset, _ in calls[:-1])
    assert test_set is not val_set
    assert not {tuple(c) for c in val_set.coords.tolist()} & {tuple(c) for c in test_set.coords.tolist()}
    for name, tensor in checkpoint["model_state"].items():
        torch.testing.assert_close(calls[-1][1][name], tensor, rtol=0, atol=0)


def test_dw6_evaluate_restores_band_removal_and_split_settings(make_scene, tmp_path):
    extra = ["--remove_water_bands", "--train_size", "0.4", "--val_size", "0.2"]
    data_path, gt_path, checkpoint, metrics = run_train(
        make_scene, tmp_path, bands=220, epochs=2, seed=5, extra=extra)
    assert checkpoint["config"]["remove_water_bands"] is True

    np.random.seed(123)  # unrelated global random state must not change the features
    np.random.rand(1000)
    result = evaluate_script.main([
        "--dataset", "IP", "--checkpoints_dir", str(tmp_path),
        "--data_path", str(data_path), "--gt_path", str(gt_path), "--device", "cpu",
    ])

    assert result["OA"] == float(metrics["test_oa"])
    np.testing.assert_array_equal(result["confusion_matrix"], metrics["test_confusion_matrix"])


def test_dw6_checkpoint_without_settings_rejected(tmp_path):
    path = tmp_path / "old.pth"
    torch.save(HSSNB(num_classes=3).state_dict(), path)
    with pytest.raises(ValueError, match="no stored settings"):
        evaluate_script.load_checkpoint(path)


def test_dw4_best_epoch_is_kept_when_a_later_epoch_is_worse(make_scene, tmp_path, monkeypatch):
    scripted_val_oa = iter([0.5, 0.9, 0.7])
    states = []

    def scripted_evaluate(model, criterion, loader, device, num_classes):
        loss, metrics = evaluate(model, criterion, loader, device, num_classes)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        return loss, {**metrics, "OA": next(scripted_val_oa, metrics["OA"])}

    monkeypatch.setattr(train, "evaluate", scripted_evaluate)
    _, _, checkpoint, metrics = run_train(make_scene, tmp_path, epochs=3)

    assert checkpoint["best_epoch"] == 2
    np.testing.assert_array_equal(metrics["val_oa"], [0.5, 0.9, 0.7])
    for name, tensor in checkpoint["model_state"].items():
        torch.testing.assert_close(states[1][name], tensor, rtol=0, atol=0)
    # The last epoch's weights differ, so keeping epoch 2 was a real choice.
    assert any(not torch.equal(states[2][k], v) for k, v in checkpoint["model_state"].items())


@pytest.mark.parametrize("val_size", ["0", "0.01"], ids=["zero", "rounds-to-zero"])
def test_empty_validation_split_rejected_before_training(make_scene, tmp_path, val_size):
    with pytest.raises(ValueError, match="Empty split"):
        run_train(make_scene, tmp_path, epochs=1, extra=["--val_size", val_size])
    assert not (tmp_path / "best_model_IP.pth").exists()


def test_zero_epochs_rejected_even_with_an_old_checkpoint(make_scene, tmp_path):
    run_train(make_scene, tmp_path, epochs=1)
    with pytest.raises(SystemExit):
        run_train(make_scene, tmp_path, epochs=0)
