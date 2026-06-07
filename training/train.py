"""Blur detection model training script.

Usage:
    python train.py --data-dir data/datasets --epochs 100
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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
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
            transforms.Resize((cfg.image_size + 16, cfg.image_size + 16)),
            transforms.RandomCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(cfg: Config):
    """Create train, val dataloaders."""
    train_dir = os.path.join(cfg.dataset_root, "train")
    val_dir = os.path.join(cfg.dataset_root, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data not found at {train_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(cfg, True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(cfg, False))

    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    import platform
    num_workers = 0 if platform.system() == "Darwin" else cfg.num_workers

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
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


@torch.no_grad()
def validate_with_tta(model, dataloader, criterion, device, num_tta=5):
    """Test-Time Augmentation: average predictions over multiple augmented views."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    tta_transforms = [
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((272, 272)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]

    # Get original dataset and apply TTA
    original_dataset = dataloader.dataset
    all_probs = []
    all_labels = []

    for tta_idx in range(min(num_tta, len(tta_transforms))):
        # Create a temporary dataset with TTA transform
        tta_dataset = datasets.ImageFolder(original_dataset.root, transform=tta_transforms[tta_idx])
        tta_loader = DataLoader(tta_dataset, batch_size=dataloader.batch_size,
                               shuffle=False, num_workers=dataloader.num_workers,
                               pin_memory=True)

        probs_list = []
        labels_list = []
        for inputs, labels in tta_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            probs_list.append(probs.cpu())
            labels_list.append(labels)

        all_probs.append(torch.cat(probs_list))
        if tta_idx == 0:
            all_labels = torch.cat(labels_list)

    # Average predictions across TTA views
    avg_probs = torch.stack(all_probs).mean(dim=0)
    _, predicted = avg_probs.max(1)
    total = len(all_labels)
    correct = predicted.eq(all_labels).sum().item()

    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train blur detection model")
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--backbone", default="mobilenet_v3_large",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tta", action="store_true", help="Use TTA for validation")
    args = parser.parse_args()

    cfg = Config(
        dataset_root=args.data_dir, backbone=args.backbone,
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, dropout=args.dropout,
        image_size=args.image_size, seed=args.seed,
        output_dir=args.output_dir, weight_decay=args.weight_decay,
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

    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = build_model(
        num_classes=cfg.num_classes, backbone=cfg.backbone,
        pretrained=cfg.pretrained, dropout=cfg.dropout,
    ).to(device)

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"  Epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.backbone} ({total_params:,} params)")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

    # Cosine Annealing with Warm Restarts - more stable than OneCycleLR
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nConfig: epochs={cfg.num_epochs}, lr={cfg.learning_rate}, "
          f"bs={cfg.batch_size}, img={cfg.image_size}")
    print(f"        label_smoothing={args.label_smoothing}, "
          f"weight_decay={cfg.weight_decay}, patience={args.patience}")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        if args.tta and (epoch + 1) % 5 == 0:
            val_acc_tta = validate_with_tta(model, val_loader, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"  Val Acc (standard): {val_acc:.4f}  (TTA): {val_acc_tta:.4f}")
        else:
            val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

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

        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Save final
    final_path = os.path.join(cfg.output_dir, f"{cfg.model_save_name}_final.pth")
    torch.save({"epoch": cfg.num_epochs, "model_state_dict": model.state_dict(),
                "config": vars(args)}, final_path)

    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model: {cfg.output_dir}/{cfg.model_save_name}_best.pth")

    # Final TTA evaluation
    if args.tta:
        print("\nRunning final TTA evaluation...")
        best_ckpt = torch.load(os.path.join(cfg.output_dir, f"{cfg.model_save_name}_best.pth"),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        tta_acc = validate_with_tta(model, val_loader, criterion, device, num_tta=5)
        print(f"Final TTA accuracy: {tta_acc:.4f}")


if __name__ == "__main__":
    main()
