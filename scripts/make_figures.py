"""Draw the README figures from trained checkpoints.

1. results_map_<theme>.png: Indian Pines, which pixels the model learned from and
   where its test predictions are wrong.
2. per_class_<theme>.png: test accuracy of every class in each scene.

Each figure is drawn for a light and a dark page. Run from the repository root:

    python scripts/make_figures.py --device mps

It reads data/ and checkpoints/ in the layout the README describes and writes to
docs/images/. It also prints a Markdown table of the per-class accuracies.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_checkpoint  # noqa: E402
from hssnb.preprocessor import HSIPreprocessor  # noqa: E402
from hssnb.training import resolve_device  # noqa: E402

SCENES = {
    "IP": ("Indian Pines", "ip/Indian_pines_corrected.mat", "ip/Indian_pines_gt.mat"),
    "PU": ("Pavia University", "pu/PaviaU.mat", "pu/PaviaU_gt.mat"),
    "SA": ("Salinas", "sa/Salinas_corrected.mat", "sa/Salinas_gt.mat"),
}

# Class names in label order, as published with the scenes.
CLASS_NAMES = {
    "IP": ["Alfalfa", "Corn, no till", "Corn, minimum till", "Corn", "Grass, pasture",
           "Grass, trees", "Grass, pasture mowed", "Hay, windrowed", "Oats", "Soybean, no till",
           "Soybean, minimum till", "Soybean, clean", "Wheat", "Woods",
           "Buildings, grass, trees, drives", "Stone and steel towers"],
    "PU": ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare soil",
           "Bitumen", "Self-blocking bricks", "Shadows"],
    "SA": ["Broccoli green weeds 1", "Broccoli green weeds 2", "Fallow", "Fallow, rough plough",
           "Fallow, smooth", "Stubble", "Celery", "Grapes, untrained", "Vineyard soil, developing",
           "Corn, senesced, green weeds", "Lettuce romaine, 4 weeks", "Lettuce romaine, 5 weeks",
           "Lettuce romaine, 6 weeks", "Lettuce romaine, 7 weeks", "Vineyard, untrained",
           "Vineyard, vertical trellis"],
}

# Colours from a palette checked for colour-blind separation and contrast in each mode.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", neutral="#d3d2ca", train="#2a78d6", val="#eb6834",
                  wrong="#e34948", dot="#2a78d6", gray="gray"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", neutral="#4a4945", train="#3987e5", val="#d95926",
                 wrong="#e66767", dot="#3987e5", gray="gray"),
}

FONT = ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def predict_test(checkpoint_path, data_path, gt_path, device, batch_size=256):
    """Rebuild the checkpoint's split and return (preprocessor, split, predicted test labels)."""
    model, config = load_checkpoint(checkpoint_path)
    model.to(device)
    pre = HSIPreprocessor(data_path, gt_path, config["dataset"], config["n_components"],
                          config["window_size"], config["remove_water_bands"])
    split = pre.preprocess_pipeline(config["train_size"], config["val_size"], config["seed"],
                                    split=config.get("split", "random"),
                                    block_size=config.get("block_size"))
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(split["test"], batch_size=batch_size):
            preds.append(model(x.to(device)).argmax(dim=1).cpu())
    return pre, split, torch.cat(preds).numpy()


def training_pixels_in_windows(shape, train_coords, test_coords, window_size):
    """Number of training pixels inside the window around each test pixel."""
    mask = np.zeros(shape, dtype=np.int64)
    mask[train_coords[:, 0], train_coords[:, 1]] = 1
    half = window_size // 2
    padded = np.pad(mask, half)
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r, c = test_coords[:, 0], test_coords[:, 1]
    s = window_size
    return integral[r + s, c + s] - integral[r, c + s] - integral[r + s, c] + integral[r, c]


def _style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["grid"],
        "xtick.color": t["muted"], "ytick.color": t["ink2"], "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
    })


def _panel_title(ax, number, title, subtitle, t):
    ax.set_title(f"{number}  {title}", loc="left", fontsize=12, fontweight="bold", color=t["ink"],
                 pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=t["ink2"], va="bottom")


def plot_map(pre, split, y_pred, path, theme, scene_name="Indian Pines", band=46):
    """Three panels: the scene, which pixels trained the model, and where test predictions are wrong."""
    t = THEMES[theme]
    _style(t)
    h, w = pre.gt.shape
    test = split["test"]
    wrong = y_pred != np.asarray(test.labels)
    n_test, n_wrong = len(wrong), int(wrong.sum())
    window = test.window_size if hasattr(test, "window_size") else 25

    roles = np.zeros((h, w), dtype=int)  # 0 unlabelled, 1 train, 2 validation, 3 test
    for code, name in ((1, "train"), (2, "val"), (3, "test")):
        rc = np.asarray(split[name].coords)
        roles[rc[:, 0], rc[:, 1]] = code
    counts = training_pixels_in_windows((h, w), np.asarray(split["train"].coords),
                                        np.asarray(test.coords), window)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.71, bottom=0.1, wspace=0.08)
    fig.suptitle(f"{scene_name}: {100 * (1 - n_wrong / n_test):.2f}% of test pixels classified "
                 f"correctly", x=0.02, y=0.97, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.9, "Every labelled pixel is classified from the 25×25-pixel window around it. "
             "Test pixels are never trained on,\nbut with a random split their windows overlap "
             "training pixels, which is one reason the score is so high.",
             fontsize=10, color=t["ink2"], va="top")

    # 1: the scene, one near-infrared band, contrast-stretched.
    img = pre.data[..., min(band, pre.data.shape[-1] - 1)].astype(float)
    lo, hi = np.percentile(img, (2, 98))
    axes[0].imshow(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), cmap=t["gray"],
                   interpolation="nearest")
    _panel_title(axes[0], "1", "The scene", f"One of its {pre.data.shape[-1]} spectral bands "
                 "(near infrared)", t)

    # 2: which pixels the model learned from, with one test window outlined.
    cmap = ListedColormap([t["surface"], t["train"], t["val"], t["neutral"]])
    axes[1].imshow(roles, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    median = np.median(counts)
    rows, cols = np.asarray(test.coords).T
    # Outline a window whose training-pixel count is the median, as near the centre as possible.
    near_median = np.abs(counts - median) <= 2
    from_centre = np.hypot(rows - h / 2, cols - w / 2)
    k = int(np.argmin(np.where(near_median, from_centre, np.inf)))
    r, c = rows[k], cols[k]
    axes[1].add_patch(Rectangle((c - window / 2, r - window / 2), window, window, fill=False,
                                edgecolor=t["ink"], linewidth=1.5))
    axes[1].plot(c, r, marker="+", color=t["ink"], markersize=9, markeredgewidth=1.5)
    axes[1].annotate(f"A typical test pixel (+):\n{int(counts[k])} of the {window * window} pixels\n"
                     "in its window are training pixels", xy=(c + window / 2, r), xytext=(6, 0),
                     textcoords="offset points", fontsize=9, color=t["ink"], va="center",
                     bbox=dict(boxstyle="round,pad=0.35", facecolor=t["surface"],
                               edgecolor=t["grid"], alpha=0.95))
    _panel_title(axes[1], "2", "What the model learned from", "Labelled pixels, split at random", t)
    n = {name: len(split[name]) for name in ("train", "val", "test")}
    axes[1].legend(handles=[
        Patch(facecolor=t["train"], label=f"Training  {n['train']:,}"),
        Patch(facecolor=t["val"], label=f"Validation  {n['val']:,}"),
        Patch(facecolor=t["neutral"], label=f"Test  {n['test']:,}"),
    ], loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, frameon=False, fontsize=9,
        handlelength=1.2, columnspacing=1.2, labelcolor=t["ink2"])

    # 3: test outcome; wrong pixels drawn as large rings so single pixels stay visible.
    outcome = np.where(roles == 3, 1, 0)
    axes[2].imshow(outcome, cmap=ListedColormap([t["surface"], t["neutral"]]), vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].scatter(cols[wrong], rows[wrong], s=46, facecolor=t["wrong"], edgecolor=t["surface"],
                    linewidth=1.2, zorder=3)
    _panel_title(axes[2], "3", "Where the test predictions are wrong",
                 f"{n_wrong} of {n_test:,} test pixels", t)
    axes[2].legend(handles=[
        Patch(facecolor=t["neutral"], label=f"Correct  {n_test - n_wrong:,}"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=t["wrong"],
               markeredgecolor=t["surface"], label=f"Wrong  {n_wrong}"),
    ], loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False, fontsize=9,
        handlelength=1.2, columnspacing=1.2, labelcolor=t["ink2"])

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"n_test": n_test, "n_wrong": n_wrong, "median_train_in_window": float(median)}


def per_class_accuracy(y_true, y_pred, num_classes):
    """Return (accuracy, support) per class; accuracy is NaN for a class with no test pixels."""
    support = np.bincount(y_true, minlength=num_classes)
    right = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return right / support, support


def plot_per_class(results, path, theme):
    """Dot plot of per-class test accuracy, one panel per scene, on a shared accuracy axis.

    results: list of (scene name, class names, accuracy array, support array, overall accuracy).
    """
    t = THEMES[theme]
    _style(t)
    lowest = min(np.nanmin(acc) for _, _, acc, _, _ in results)
    x_lo = min(98.0, np.floor(lowest * 100) - 0.5)
    heights = [len(names) for _, names, _, _, _ in results]
    fig, axes = plt.subplots(len(results), 1, figsize=(8.6, 0.25 * sum(heights) + 3.2),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.3})
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.36, right=0.93, top=1 - 1.6 / fig.get_figheight(), bottom=0.04)
    fig.suptitle("Test accuracy for every class", x=0.02, y=1 - 0.25 / fig.get_figheight(),
                 ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.02, 1 - 0.68 / fig.get_figheight(),
             "Share of each class's test pixels classified correctly. The number after each name is "
             "its test-pixel count;\nsmall classes move most when a few pixels are wrong. Values "
             "below 100% are labelled.", fontsize=10, color=t["ink2"], va="top")

    for ax, (scene, names, acc, support, oa) in zip(axes, results):
        y = np.arange(len(names))[::-1]
        for yy in y:
            ax.axhline(yy, color=t["grid"], linewidth=0.8, zorder=0)
        ax.axvline(oa * 100, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.text(oa * 100, len(names) - 0.35, f"overall {oa * 100:.2f}%", fontsize=8.5,
                color=t["ink2"], ha="right", va="bottom")
        ax.scatter(acc * 100, y, s=52, color=t["dot"], edgecolor=t["surface"], linewidth=1.2, zorder=3)
        for yy, a in zip(y, acc):
            if a < 1:
                label = f"{a * 100:.1f}%" if round(a * 100, 1) < 100 else f"{a * 100:.2f}%"
                ax.text(a * 100 - 0.12, yy, label, fontsize=8.5, color=t["ink2"],
                        ha="right", va="center")
        ax.set_yticks(y, [f"{name}  ({s:,})" for name, s in zip(names, support)], fontsize=9)
        ax.set_title(scene, loc="left", fontsize=12, fontweight="bold", color=t["ink"], x=-0.53)
        ax.set_xlim(x_lo, 100.4)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.tick_params(axis="both", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown_table(results):
    lines = []
    for scene, names, acc, support, oa in results:
        lines += [f"**{scene}** (overall {oa * 100:.2f}%)", "", "| Class | Test pixels | Accuracy |",
                  "| --- | ---: | ---: |"]
        lines += [f"| {name} | {s:,} | {a * 100:.2f}% |" for name, a, s in zip(names, acc, support)]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--out_dir", default=os.path.join("docs", "images"))
    parser.add_argument("--device", default="cuda", help='"cpu", "cuda" or "mps"')
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for key, (scene, data_file, gt_file) in SCENES.items():
        pre, split, y_pred = predict_test(os.path.join(args.checkpoints_dir, f"best_model_{key}.pth"),
                                          os.path.join(args.data_dir, data_file),
                                          os.path.join(args.data_dir, gt_file), device)
        y_true = np.asarray(split["test"].labels)
        acc, support = per_class_accuracy(y_true, y_pred, len(CLASS_NAMES[key]))
        results.append((scene, CLASS_NAMES[key], acc, support, float(np.mean(y_true == y_pred))))
        if key == "IP":
            for theme in THEMES:
                stats = plot_map(pre, split, y_pred, os.path.join(args.out_dir, f"results_map_{theme}.png"),
                                 theme, scene)
            print(f"IP map: {stats}")
    for theme in THEMES:
        plot_per_class(results, os.path.join(args.out_dir, f"per_class_{theme}.png"), theme)
    print(markdown_table(results))


if __name__ == "__main__":
    main()
