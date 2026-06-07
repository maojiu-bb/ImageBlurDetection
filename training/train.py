"""Blur detection model training script.

Usage:
    python train.py --config model/config.py
    python train.py --data-dir data/datasets --epochs 50 --lr 1e-3
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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
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
    """Get data transforms for training or validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=cfg.random_horizontal_flip),
            transforms.RandomRotation(cfg.random_rotation),
            transforms.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
    # MPS backend doesn't benefit much from multiprocessing data loading
    import platform
    num_workers = 0 if platform.system() == "Darwin" else cfg.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
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
    parser.add_argument("--backbone", default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
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

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    if cfg.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs)
    else:
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    # Warmup: linear lr increase for first few epochs
    warmup_scheduler = None
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )

    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nStarting training for {cfg.num_epochs} epochs (from epoch {start_epoch})...")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        if warmup_scheduler and epoch < cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

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
