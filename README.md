# HSSNB

A PyTorch implementation of the hybrid CNN and Bi-LSTM network for hyperspectral
image classification described in Sahoo and Chakraborty,
[Hybrid CNN Bi-LSTM neural network for Hyperspectral image classification](https://arxiv.org/abs/2402.10026)
(2024). It covers the paper's three benchmark scenes: Indian Pines (`IP`),
Pavia University (`PU`) and Salinas (`SA`).

The network classifies one pixel at a time from the square window of pixels
around it. 3D convolutions read spectral and spatial patterns together, 2D
convolutions refine the spatial features, and two bidirectional LSTM layers read
the resulting feature map as a sequence of positions.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e ".[notebooks]"` adds Jupyter and the packages that the notebooks
in `notebooks/` use.

## Data

Download the scenes from the
[Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
page (Grupo de Inteligencia Computacional, UPV/EHU). Each scene comes as two
MATLAB files, the image and its ground truth, and each file holds a single
array. The scripts take the file paths as arguments. The examples below assume
this layout, which `.gitignore` keeps out of git:

```
data/
  ip/  Indian_pines_corrected.mat   Indian_pines_gt.mat
  pu/  PaviaU.mat                   PaviaU_gt.mat
  sa/  Salinas_corrected.mat        Salinas_gt.mat
```

The "corrected" Indian Pines and Salinas files already leave out the
water-absorption bands. `--remove_water_bands` is only for the raw files
(`Indian_pines.mat`, 220 bands, and `Salinas.mat`, 224 bands). It drops the same
20 bands the corrected files leave out, so both routes give 200 and 204 bands,
and it raises an error when given an already-corrected file. Pavia University
needs no band removal.

## Train

```bash
python train.py --dataset IP \
  --data_path data/ip/Indian_pines_corrected.mat --gt_path data/ip/Indian_pines_gt.mat
```

The scene is standardised band by band and reduced to `--n_components`
principal components (default 30). Every labelled pixel becomes one sample, a
`--window_size` square (default 25) centred on it, with reflected borders. The
labelled pixels of each class are split at random into 30% training
(`--train_size`), 10% validation (`--val_size`) and the rest, 60%, for testing.

Each epoch trains on the training split and scores the validation split. The
epoch with the best validation overall accuracy is saved to
`checkpoints/best_model_<dataset>.pth`, together with the settings needed to
rebuild the model and the split. After the last epoch, that checkpoint is scored
once on the test split. Per-epoch losses and validation metrics, and the test
metrics, go to `checkpoints/metrics_<dataset>.npz`.

`--seed` (default 42) fixes the split, the weight initialisation and the batch
order, so a CPU run repeats exactly. Run `python train.py --help` for all
options.

## Evaluate

```bash
python evaluate.py --dataset IP \
  --data_path data/ip/Indian_pines_corrected.mat --gt_path data/ip/Indian_pines_gt.mat
```

This loads the checkpoint, rebuilds the same test split from the stored
settings, and prints overall accuracy (OA), average per-class accuracy (AA) and
Cohen's kappa. It writes the loss and validation-metric curves and the test
confusion matrix to `checkpoints/eval_results/<dataset>/`.

## Model

Shapes for the default 30 components and 25 × 25 window; batch dimension left
out.

| Layer | Output shape |
|---|---|
| Input patch | 1 × 30 × 25 × 25 |
| Conv3d, 8 filters, kernel 7 × 3 × 3, ReLU | 8 × 24 × 23 × 23 |
| Conv3d, 16 filters, kernel 5 × 3 × 3, ReLU | 16 × 20 × 21 × 21 |
| Conv3d, 32 filters, kernel 3 × 3 × 3, ReLU | 32 × 18 × 19 × 19 |
| Merge filters and bands | 576 × 19 × 19 |
| Conv2d, 64 filters, 3 × 3, ReLU | 64 × 17 × 17 |
| Conv2d, 128 filters, 3 × 3, ReLU | 128 × 15 × 15 |
| Positions as a sequence, LayerNorm | 225 × 128 |
| Bi-LSTM, 64 units each way, dropout 0.25 | 225 × 128 |
| Bi-LSTM, 64 units each way | 225 × 128 |
| Last position, linear layer | number of classes |

That is 626,816 trainable parameters for a 16-class scene. The layer sizes
follow `--n_components`, which must be at least 13; the window must be at least
11 pixels.

## Implementation notes

These are choices made in this code. Check them against the paper before
comparing numbers.

- The 3D kernels are written in PyTorch's (bands, height, width) order, so the
  paper's 3 × 3 × 7 kernel is `(7, 3, 3)` here.
- A LayerNorm sits between the convolutions and the first Bi-LSTM layer.
- The classifier reads the Bi-LSTM output at the last of the 225 positions.
- PCA is fitted on all pixels of the scene, labelled or not.
- The paper trains on 30% of the labelled pixels. Here 10% more are held out to
  pick the best epoch, so 60% are left for testing.

## Results

No results from this code on the real datasets have been published here yet.
The paper's abstract reports overall accuracies of 99.83% (IP), 99.98% (PU)
and 100% (SA).

## Tests

```bash
pytest -q
ruff check .
```

The tests build small synthetic scenes, so they need no downloaded data, and
run in a few seconds on a CPU. GitHub Actions runs the same two commands.

## Notebooks

`notebooks/` holds early data exploration: `data_inspection.ipynb` looks at the
raw Indian Pines scene, and `inspect_GRSS.ipynb` at the 2013 IEEE GRSS Houston
data, which the scripts do not support. Outputs are stripped; run them against
your own copy of the data.

## Citation

```bibtex
@misc{sahoo2024hybrid,
  title         = {Hybrid CNN Bi-LSTM neural network for Hyperspectral image classification},
  author        = {Sahoo, Alok Ranjan and Chakraborty, Pavan},
  year          = {2024},
  eprint        = {2402.10026},
  archivePrefix = {arXiv}
}
```
