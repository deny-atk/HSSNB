"""Train HSSNB on one scene and score the best checkpoint on the held-out test split."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hssnb.models import HSSNB
from hssnb.preprocessor import DATASETS, HSIPreprocessor
from hssnb.training import evaluate, seed_everything, train_one_epoch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train HSSNB model on Hyperspectral Data')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset .mat file')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth .mat file')
    parser.add_argument('--n_components', type=int, default=30, help='Number of PCA components')
    parser.add_argument('--window_size', type=int, default=25, help='Spatial window size')
    parser.add_argument('--remove_water_bands', action='store_true',
                        help='Remove water absorption bands (only for the raw IP and SA files)')
    parser.add_argument('--train_size', type=float, default=0.3, help='Training data proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation data proportion, used to pick the best epoch')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the split and training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args(argv)
    if args.num_epochs < 1:
        parser.error('--num_epochs must be at least 1')
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    preprocessor = HSIPreprocessor(
        data_path=args.data_path,
        gt_path=args.gt_path,
        dataset_name=args.dataset,
        n_components=args.n_components,
        window_size=args.window_size,
        remove_water_bands=args.remove_water_bands
    )
    split = preprocessor.preprocess_pipeline(
        train_size=args.train_size, val_size=args.val_size, seed=args.seed)
    num_classes = split['num_classes']
    if len(split['train']) == 0 or len(split['val']) == 0:
        raise ValueError(f"Empty split: train {len(split['train'])}, validation {len(split['val'])} "
                         f"samples. Raise --train_size or --val_size.")
    print(f"Samples: train {len(split['train'])}, validation {len(split['val'])}, "
          f"test {len(split['test'])}")

    train_loader, val_loader, test_loader = (
        DataLoader(split[name], batch_size=args.batch_size, shuffle=(name == 'train'),
                   num_workers=args.num_workers)
        for name in ('train', 'val', 'test'))

    model = HSSNB(num_classes=num_classes, n_components=args.n_components).to(device)
    print(f"Total trainable parameters: {model.count_parameters()}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Everything evaluate.py needs to rebuild the model and the same test split.
    config = {
        'dataset': args.dataset, 'n_components': args.n_components,
        'window_size': args.window_size, 'remove_water_bands': args.remove_water_bands,
        'train_size': args.train_size, 'val_size': args.val_size, 'seed': args.seed,
        'num_classes': num_classes,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"best_model_{args.dataset}.pth")

    best_oa = -1.0
    history = {k: [] for k in ('train_losses', 'val_losses', 'val_oa', 'val_aa', 'val_kappa')}

    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, metrics = evaluate(model, criterion, val_loader, device, num_classes)
        elapsed_time = time.time() - start_time

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['val_oa'].append(metrics['OA'])
        history['val_aa'].append(metrics['AA'])
        history['val_kappa'].append(metrics['Kappa'])

        print(f"Epoch [{epoch}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val OA: {metrics['OA']*100:.2f}% "
              f"Val AA: {metrics['AA']*100:.2f}% "
              f"Val Kappa: {metrics['Kappa']:.4f} "
              f"Time: {elapsed_time:.2f}s")

        # Keep the epoch with the best validation OA; the test split is not looked at here.
        if metrics['OA'] > best_oa:
            best_oa = metrics['OA']
            torch.save({'model_state': model.state_dict(), 'config': config, 'best_epoch': epoch},
                       checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Training complete.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    _, test = evaluate(model, criterion, test_loader, device, num_classes)
    print(f"Test (best epoch {checkpoint['best_epoch']}): OA {test['OA']*100:.2f}% "
          f"AA {test['AA']*100:.2f}% Kappa {test['Kappa']:.4f}")

    metrics_path = os.path.join(args.save_dir, f"metrics_{args.dataset}.npz")
    np.savez(metrics_path, **history, test_oa=test['OA'], test_aa=test['AA'],
             test_kappa=test['Kappa'], test_confusion_matrix=test['confusion_matrix'])
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
