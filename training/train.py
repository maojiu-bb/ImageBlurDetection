"""Blur detection model training script.

Usage:
    python train.py --data-dir data/datasets --epochs 100 --backbone mobilenet_v3_large

Key improvements for high accuracy:
    - Label smoothing for better generalization
    - Mixup augmentation for regularization
    - More aggressive data augmentation (RandomAffine, GaussianBlur, RandomErasing)
    - OneCycleLR scheduler for faster convergence
    - Gradient clipping for training stability
    - Larger model backbone option (mobilenet_v3_large)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
from torchvision import datasets, transforms
from tqdm import tqdm

from model.config import Config
from model.network import build_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(cfg: Config, is_train: bool):
    """Get data transforms for training or validation.

    Training augmentation pipeline includes:
    - Random resized crop with scale variation
    - Random horizontal/vertical flips
    - Random rotation
    - Color jitter (brightness, contrast, saturation, hue)
    - Random affine transforms
    - Random perspective distortion
    - Gaussian blur augmentation
    - Random erasing (cutout)
    - ImageNet normalization
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size + 32, cfg.image_size + 32)),
            transforms.RandomCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(p=cfg.random_horizontal_flip),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(cfg.random_rotation),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(cfg: Config):
    """Create train, val, test dataloaders."""
    train_dir = os.path.join(cfg.dataset_root, "train")
    val_dir = os.path.join(cfg.dataset_root, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_dir}\n"
            f"Run 'python data/prepare_dataset.py --source <sharp_images> --output {cfg.dataset_root}' first."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(cfg, True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(cfg, False))

    print(f"Classes: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # On macOS, use 0 workers to avoid "Too many open files" errors
    import platform
    num_workers = 0 if platform.system() == "Darwin" else cfg.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation.

    Mixup creates virtual training examples by linearly interpolating
    between pairs of input images and their labels.

    Args:
        x: Input batch tensor.
        y: Label batch tensor.
        alpha: Mixup interpolation parameter (higher = more mixing).

    Returns:
        Mixed inputs, original labels, shuffled labels, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scheduler=None, use_mixup=True, mixup_alpha=0.2, grad_clip=1.0):
    """Train for one epoch with optional mixup and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping for training stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Step scheduler per batch (required for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # For mixup, use original labels for accuracy tracking
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="  Validating"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train blur detection model")
    parser.add_argument("--data-dir", default="data/datasets",
                        help="Dataset root directory")
    parser.add_argument("--backbone", default="mobilenet_v3_large",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Mixup alpha parameter (default: 0.2, 0 to disable)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    args = parser.parse_args()

    # Configuration
    cfg = Config(
        dataset_root=args.data_dir,
        backbone=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        image_size=args.image_size,
        seed=args.seed,
        output_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.patience,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = build_model(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
    ).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_acc = checkpoint.get("val_acc", 0.0)
            print(f"  Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
        else:
            print(f"Warning: checkpoint not found at {args.resume}, starting from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.backbone}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"Label smoothing: {args.label_smoothing}")

    # Optimizer with higher learning rate for larger model
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    # OneCycleLR scheduler - more aggressive but effective
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,  # initial_lr = max_lr / 25
        final_div_factor=1000,  # final_lr = initial_lr / 1000
    )

    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nTraining config:")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Mixup alpha: {args.mixup_alpha}")
    print(f"  Gradient clip: {args.grad_clip}")
    print(f"  Patience: {cfg.early_stopping_patience}")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler,
            use_mixup=args.mixup_alpha > 0,
            mixup_alpha=args.mixup_alpha,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg.output_dir, f"{cfg.model_save_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": vars(args),
            }, save_path)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Save final model
    final_path = os.path.join(cfg.output_dir, f"{cfg.model_save_name}_final.pth")
    torch.save({
        "epoch": cfg.num_epochs,
        "model_state_dict": model.state_dict(),
        "config": vars(args),
    }, final_path)

    # Save training history
    history_path = os.path.join(cfg.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model: {cfg.output_dir}/{cfg.model_save_name}_best.pth")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
