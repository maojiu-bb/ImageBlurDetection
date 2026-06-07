"""Dataset preparation for blur detection training.

Generates synthetic blurry images from sharp source images by applying
motion blur, defocus blur, and Gaussian blur kernels.

Each source image generates multiple blur variants with different parameters
and severity levels, plus augmented sharp images (flips, rotations).

Usage:
    python prepare_dataset.py --source <sharp_images_dir> --output <output_dir> --variants 10

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


def augment_sharp(img_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentation to a sharp image.

    Applies random horizontal flip, vertical flip, and 90-degree rotations.
    This creates more diverse sharp image samples without changing semantics.

    Args:
        img_np: Input image as numpy array (H, W, C).
        rng: Random number generator.

    Returns:
        Augmented image as numpy array.
    """
    img = img_np.copy()

    # Random horizontal flip
    if rng.random() < 0.5:
        img = np.flip(img, axis=1).copy()

    # Random vertical flip
    if rng.random() < 0.3:
        img = np.flip(img, axis=0).copy()

    # Random 90-degree rotations
    k = rng.integers(0, 4)
    if k > 0:
        img = np.rot90(img, k=k).copy()

    return img


def generate_blur_variants(image_path: str, output_dir: str, image_size: int = 256,
                           num_variants: int = 10):
    """Generate multiple blur variants for a single sharp image.

    Creates num_variants variants per blur type with different random parameters
    and severity levels. Also generates augmented sharp images.

    Args:
        image_path: Path to the sharp source image.
        output_dir: Root output directory (contains train/val/test split dirs).
        image_size: Resize images to this size before applying blur.
        num_variants: Number of blur variants to generate per blur type.

    Returns:
        Tuple of (dict of class_name -> list of numpy arrays, stem name).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_np = np.array(img)
    except Exception as e:
        print(f"  Skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    rng = np.random.default_rng()

    result = {
        "sharp": [],
        "motion_blur": [],
        "defocus_blur": [],
        "gaussian_blur": [],
    }

    # Generate augmented sharp images (original + augmented variants)
    # Use same number as blur variants to keep classes balanced
    for i in range(num_variants):
        if i == 0:
            # Keep the original as-is
            result["sharp"].append(img_np)
        else:
            result["sharp"].append(augment_sharp(img_np, rng))

    # Generate multiple motion blur variants with different parameters
    for i in range(num_variants):
        severity = rng.choice(["light", "medium", "heavy"], p=[0.3, 0.4, 0.3])
        params = random_blur_params("motion", severity)
        kernel = motion_blur_kernel(params["size"], params["angle"])
        motion_img = apply_kernel(img_np, kernel)
        result["motion_blur"].append(motion_img)

    # Generate multiple defocus blur variants
    for i in range(num_variants):
        severity = rng.choice(["light", "medium", "heavy"], p=[0.3, 0.4, 0.3])
        params = random_blur_params("defocus", severity)
        kernel = defocus_blur_kernel(params["radius"])
        defocus_img = apply_kernel(img_np, kernel)
        result["defocus_blur"].append(defocus_img)

    # Generate multiple Gaussian blur variants
    for i in range(num_variants):
        severity = rng.choice(["light", "medium", "heavy"], p=[0.3, 0.4, 0.3])
        params = random_blur_params("gaussian", severity)
        kernel = gaussian_blur_kernel(params["size"], params["sigma"])
        gauss_img = apply_kernel(img_np, kernel)
        result["gaussian_blur"].append(gauss_img)

    return result, stem


def save_variants(variants_list: list, split_dir: str, stem: str, class_name: str):
    """Save a list of image variants to their class directory."""
    class_dir = os.path.join(split_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    for i, img_np in enumerate(variants_list):
        if i == 0:
            out_path = os.path.join(class_dir, f"{stem}.png")
        else:
            out_path = os.path.join(class_dir, f"{stem}_v{i:02d}.png")
        Image.fromarray(img_np).save(out_path)


def prepare_dataset(source_dir: str, output_dir: str,
                    image_size: int = 256, num_variants: int = 10,
                    train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Prepare the full blur detection dataset.

    Each source image generates num_variants blur variants per blur type,
    plus augmented sharp images. This multiplies the effective dataset size
    significantly.

    Args:
        source_dir: Directory containing sharp source images.
        output_dir: Output directory for the dataset.
        image_size: Resize target size.
        num_variants: Number of blur variants per image per blur type.
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

    n_source = len(source_images)
    # Each source generates: num_variants sharp + num_variants×3 blur variants
    total_per_img = num_variants * 4
    estimated_total = n_source * total_per_img
    print(f"Found {n_source} source images")
    print(f"Generating {num_variants} variants per class per image")
    print(f"Estimated total dataset size: ~{estimated_total} images")

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
        print(f"\nProcessing {split_name} split ({len(split_indices)} source images)...")
        for idx in tqdm(split_indices, desc=f"  {split_name}"):
            img_path = source_images[idx]
            result = generate_blur_variants(img_path, output_dir, image_size, num_variants)
            if result:
                variants_dict, stem = result
                split_dir = os.path.join(output_dir, split_name)
                for class_name, img_list in variants_dict.items():
                    save_variants(img_list, split_dir, stem, class_name)

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
    parser.add_argument("--variants", type=int, default=10,
                        help="Number of blur variants per image per type (default: 10)")
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
        image_size=args.image_size,
        num_variants=args.variants,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
