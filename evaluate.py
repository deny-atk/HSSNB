"""Re-score a trained HSSNB checkpoint on its test split and plot the results."""

import argparse
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hssnb.models import HSSNB
from hssnb.preprocessor import DATASETS, HSIPreprocessor
from hssnb.training import evaluate, resolve_device

matplotlib.use('Agg')  # only writes image files
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                          cmap='Blues', save_path=None):
    """
    Plots and optionally saves the confusion matrix using seaborn heatmap.
    Args:
        cm (np.ndarray): Confusion matrix of shape (num_classes, num_classes)
        class_names (list): Class labels for the axes
        title (str): Title for the heatmap
        cmap (str): Colormap to use
        save_path (str): Path to save the plot image. If None, the plot is not saved.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_curves(metrics_npz_path, save_dir):
    """
    Loads and plots training/validation losses and validation OA/AA/Kappa curves
    from the metrics .npz file, then saves them as images.
    Args:
        metrics_npz_path (str): Path to the metrics_<dataset>.npz file
        save_dir (str): Directory where images will be saved
    """
    metrics_data = np.load(metrics_npz_path)
    epochs = np.arange(1, len(metrics_data['train_losses']) + 1)

    # Plot training & validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, metrics_data['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics_data['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    plt.close()

    # Plot validation OA, AA, Kappa
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, metrics_data['val_oa'], label='OA')
    plt.plot(epochs, metrics_data['val_aa'], label='AA')
    plt.plot(epochs, metrics_data['val_kappa'], label='Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation OA, AA & Kappa Over Epochs')
    plt.legend()
    plt.tight_layout()
    metrics_plot_path = os.path.join(save_dir, 'metrics_curve.png')
    plt.savefig(metrics_plot_path)
    print(f"OA, AA & Kappa plot saved to {metrics_plot_path}")
    plt.close()


def load_checkpoint(checkpoint_path):
    """Return the trained model and the training settings stored with it."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'config' not in checkpoint:
        raise ValueError(f"{checkpoint_path} has no stored settings; retrain it with the current train.py")
    config = checkpoint['config']
    model = HSSNB(num_classes=config['num_classes'], n_components=config['n_components'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate and Visualize HSSNB Results')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS,
                        help='Dataset name; selects best_model_<dataset>.pth.')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory where model checkpoints and metrics are saved.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the HSI .mat file used for training.')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to the ground truth .mat file used for training.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation ("cpu", "cuda" or "mps"); falls back to the CPU.')
    args = parser.parse_args(argv)
    device = resolve_device(args.device)

    checkpoint_path = os.path.join(args.checkpoints_dir, f"best_model_{args.dataset}.pth")
    metrics_path = os.path.join(args.checkpoints_dir, f"metrics_{args.dataset}.npz")
    for path in (checkpoint_path, metrics_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    eval_results_dir = os.path.join(args.checkpoints_dir, 'eval_results', args.dataset)
    os.makedirs(eval_results_dir, exist_ok=True)
    plot_training_curves(metrics_path, eval_results_dir)

    model, config = load_checkpoint(checkpoint_path)
    model = model.to(device)
    print(f"Loaded {checkpoint_path}: {config}")

    # Rebuild the same test split with the settings used for training.
    preprocessor = HSIPreprocessor(
        data_path=args.data_path,
        gt_path=args.gt_path,
        dataset_name=config['dataset'],
        n_components=config['n_components'],
        window_size=config['window_size'],
        remove_water_bands=config['remove_water_bands'],
    )
    # Checkpoints from before --split existed used the random split.
    split = preprocessor.preprocess_pipeline(
        train_size=config['train_size'], val_size=config['val_size'], seed=config['seed'],
        split=config.get('split', 'random'), block_size=config.get('block_size'))
    test_loader = DataLoader(split['test'], batch_size=args.batch_size)

    num_classes = config['num_classes']
    _, metrics = evaluate(model, nn.CrossEntropyLoss(), test_loader, device, num_classes)
    print(f"Test: OA {metrics['OA']*100:.2f}% AA {metrics['AA']*100:.2f}% "
          f"Kappa {metrics['Kappa']:.4f}")

    class_names = [f"Class {i+1}" for i in range(num_classes)]
    plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                          title=f'Confusion Matrix - {args.dataset}',
                          save_path=os.path.join(eval_results_dir, 'confusion_matrix.png'))
    print(f"All plots saved to: {eval_results_dir}")
    return metrics


if __name__ == '__main__':
    main()
