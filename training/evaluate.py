"""Model evaluation script.

Generates classification report, confusion matrix, and per-class metrics.

Usage:
    python evaluate.py --model output/blur_detector_best.pth --data-dir data/datasets
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.config import Config
from model.network import build_model


def get_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def evaluate(model, dataloader, device, class_names):
    """Run evaluation and print metrics."""
    model.eval()

    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    # Per-class metrics
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)

    for i, name in enumerate(class_names):
        tp = ((all_preds == i) & (all_labels == i)).sum()
        fp = ((all_preds == i) & (all_labels != i)).sum()
        fn = ((all_preds != i) & (all_labels == i)).sum()
        support = (all_labels == i).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    # Confusion matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1

    print(f"\nConfusion Matrix:")
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"{name:>15}", end="")
        for j in range(n_classes):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # Macro-averaged metrics
    precisions, recalls, f1s = [], [], []
    for i in range(n_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    print(f"\n{'Macro Avg':<15} {np.mean(precisions):>10.4f} {np.mean(recalls):>10.4f} {np.mean(f1s):>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate blur detection model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", default="data/datasets", help="Dataset root")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone", default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {})

    # Use backbone from checkpoint if available, otherwise from args
    backbone = model_config.get("backbone", args.backbone)

    # Build model
    model = build_model(
        num_classes=4,
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Data
    data_dir = os.path.join(args.data_dir, args.split)
    dataset = datasets.ImageFolder(data_dir, transform=get_transforms(args.image_size))
    # On macOS, use 0 workers to avoid "Too many open files" errors
    import platform
    num_workers = 0 if platform.system() == "Darwin" else 4
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    class_names = dataset.classes
    print(f"Evaluating on {args.split} set ({len(dataset)} samples)")
    print(f"Classes: {class_names}")

    evaluate(model, loader, device, class_names)


if __name__ == "__main__":
    main()
