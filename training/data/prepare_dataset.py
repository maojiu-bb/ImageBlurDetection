"""Dataset preparation for blur detection training.

Generates synthetic blurry images from sharp source images by applying
motion blur, defocus blur, and Gaussian blur kernels.

Usage:
    python prepare_dataset.py --source <sharp_images_dir> --output <output_dir> --count 1000

Directory structure created:
    output/
    ├── train/
    │   ├── sharp/
    │   ├── motion_blur/
    │   ├── defocus_blur/
    │   └── gaussian_blur/
    ├── val/
    │   └── ...
    └── test/
        └── ...
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.blur_kernels import (
    motion_blur_kernel,
    defocus_blur_kernel,
    gaussian_blur_kernel,
    apply_kernel,
    random_blur_params,
)


def find_images(directory: str) -> list:
    """Find all image files in a directory recursively."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".webp"}
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))
    return sorted(images)


def generate_blur_variants(image_path: str, output_dir: str, image_size: int = 256):
    """Generate all blur variants for a single sharp image.

    Creates one variant per blur type with random parameters.

    Args:
        image_path: Path to the sharp source image.
        output_dir: Root output directory (contains train/val/test split dirs).
        image_size: Resize images to this size before applying blur.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_np = np.array(img)
    except Exception as e:
        print(f"  Skipping {image_path}: {e}")
        return

    stem = Path(image_path).stem
    rng = np.random.default_rng()

    # Motion blur
    params = random_blur_params("motion")
    kernel = motion_blur_kernel(params["size"], params["angle"])
    motion_img = apply_kernel(img_np, kernel)

    # Defocus blur
    params = random_blur_params("defocus")
    kernel = defocus_blur_kernel(params["radius"])
    defocus_img = apply_kernel(img_np, kernel)

    # Gaussian blur
    params = random_blur_params("gaussian")
    kernel = gaussian_blur_kernel(params["size"], params["sigma"])
    gauss_img = apply_kernel(img_np, kernel)

    return {
        "sharp": img_np,
        "motion_blur": motion_img,
        "defocus_blur": defocus_img,
        "gaussian_blur": gauss_img,
    }, stem


def save_split(images_dict: dict, split_dir: str, stem: str):
    """Save all blur variants to their respective class directories."""
    for class_name, img_np in images_dict.items():
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        out_path = os.path.join(class_dir, f"{stem}.png")
        Image.fromarray(img_np).save(out_path, quality=95)


def prepare_dataset(source_dir: str, output_dir: str, images_per_split: int = None,
                    image_size: int = 256, train_ratio: float = 0.7,
                    val_ratio: float = 0.15):
    """Prepare the full blur detection dataset.

    Args:
        source_dir: Directory containing sharp source images.
        output_dir: Output directory for the dataset.
        images_per_split: Limit images per split (None = use all).
        image_size: Resize target size.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (rest goes to test).
    """
    print(f"Scanning source images in {source_dir}...")
    source_images = find_images(source_dir)

    if not source_images:
        print(f"No images found in {source_dir}")
        print("\nYou can download images from:")
        print("  - COCO: https://cocodataset.org/")
        print("  - DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("  - Or any collection of sharp photographs")
        return

    print(f"Found {len(source_images)} source images")

    # Shuffle for randomness
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(source_images))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    # Create output directories
    for split_name in splits:
        for class_name in ["sharp", "motion_blur", "defocus_blur", "gaussian_blur"]:
            os.makedirs(os.path.join(output_dir, split_name, class_name), exist_ok=True)

    # Process images
    for split_name, split_indices in splits.items():
        if images_per_split:
            split_indices = split_indices[:images_per_split]

        print(f"\nProcessing {split_name} split ({len(split_indices)} images)...")
        for idx in tqdm(split_indices, desc=f"  {split_name}"):
            img_path = source_images[idx]
            result = generate_blur_variants(img_path, output_dir, image_size)
            if result:
                images_dict, stem = result
                save_split(images_dict, os.path.join(output_dir, split_name), stem)

    # Print summary
    print("\n=== Dataset Summary ===")
    for split_name in splits:
        split_dir = os.path.join(output_dir, split_name)
        total = 0
        for class_name in ["sharp", "motion_blur", "defocus_blur", "gaussian_blur"]:
            class_dir = os.path.join(split_dir, class_name)
            count = len(list(Path(class_dir).glob("*.png")))
            total += count
            print(f"  {split_name}/{class_name}: {count}")
        print(f"  {split_name}/total: {total}")
    print("========================")


def main():
    parser = argparse.ArgumentParser(description="Prepare blur detection dataset")
    parser.add_argument("--source", required=True,
                        help="Directory containing sharp source images")
    parser.add_argument("--output", default="data/datasets",
                        help="Output directory (default: data/datasets)")
    parser.add_argument("--count", type=int, default=None,
                        help="Max images per split (default: all)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Resize images to this size (default: 256)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Training split ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        images_per_split=args.count,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
