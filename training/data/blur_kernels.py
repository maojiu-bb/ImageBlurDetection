"""Blur kernel generation for synthetic data augmentation.

Generates three types of blur:
- Motion blur: linear convolution kernel simulating camera/subject motion
- Defocus blur: circular disk kernel simulating out-of-focus lens
- Gaussian blur: 2D Gaussian kernel simulating general softness
"""

import numpy as np
from typing import Tuple


def motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Create a motion blur kernel.

    Args:
        size: Length of the blur line (must be odd).
        angle: Direction of motion in degrees (0 = horizontal).

    Returns:
        Normalized 2D convolution kernel of shape (size, size).
    """
    size = max(3, size | 1)  # Ensure odd and >= 3
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    for i in range(size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center - offset * sin_a))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    # Normalize
    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def defocus_blur_kernel(radius: int) -> np.ndarray:
    """Create a defocus (disk) blur kernel.

    Simulates the circular bokeh pattern from an out-of-focus lens.

    Args:
        radius: Radius of the disk in pixels.

    Returns:
        Normalized 2D convolution kernel.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = radius

    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if dist <= radius:
                kernel[y, x] = 1.0

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def gaussian_blur_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a Gaussian blur kernel.

    Args:
        size: Kernel size (must be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Normalized 2D convolution kernel.
    """
    size = max(3, size | 1)  # Ensure odd and >= 3
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    for y in range(size):
        for x in range(size):
            dist_sq = (x - center) ** 2 + (y - center) ** 2
            kernel[y, x] = np.exp(-dist_sq / (2.0 * sigma ** 2))

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a convolution kernel to an image using scipy or manual convolution.

    Args:
        image: Input image as numpy array (H, W, C) with uint8 values.
        kernel: 2D convolution kernel.

    Returns:
        Blurred image as uint8 numpy array.
    """
    from PIL import Image
    import PIL.ImageFilter

    # Convert kernel to PIL filter format
    k_size = kernel.shape[0]
    k_flat = kernel.flatten().tolist()

    # PIL uses a different kernel format - we need to use scipy instead
    try:
        from scipy.ndimage import convolve
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve(
                image[:, :, c].astype(np.float32), kernel, mode='reflect'
            )
        return np.clip(result, 0, 255).astype(np.uint8)
    except ImportError:
        # Fallback: use PIL's built-in filters for common blur types
        pil_img = Image.fromarray(image)
        blurred = pil_img.filter(PIL.ImageFilter.GaussianBlur(radius=kernel.shape[0] // 4))
        return np.array(blurred)


def random_blur_params(blur_type: str, severity: str = "random") -> dict:
    """Generate random blur parameters for a given blur type.

    Args:
        blur_type: One of "motion", "defocus", "gaussian".
        severity: "light", "medium", "heavy", or "random".

    Returns:
        Dict of parameters for the corresponding kernel function.
    """
    rng = np.random.default_rng()

    severity_ranges = {
        "light":  {"motion_size": (3, 9),   "motion_angle": (0, 360),
                   "defocus_radius": (2, 5),  "gauss_size": (3, 7),
                   "gauss_sigma": (0.5, 2.0)},
        "medium": {"motion_size": (9, 21),  "motion_angle": (0, 360),
                   "defocus_radius": (5, 12), "gauss_size": (7, 15),
                   "gauss_sigma": (2.0, 5.0)},
        "heavy":  {"motion_size": (21, 41), "motion_angle": (0, 360),
                   "defocus_radius": (12, 25),"gauss_size": (15, 31),
                   "gauss_sigma": (5.0, 12.0)},
    }

    if severity == "random":
        severity = rng.choice(["light", "medium", "heavy"], p=[0.3, 0.4, 0.3])

    ranges = severity_ranges[severity]

    if blur_type == "motion":
        return {
            "size": int(rng.integers(*ranges["motion_size"])),
            "angle": float(rng.uniform(*ranges["motion_angle"])),
        }
    elif blur_type == "defocus":
        return {
            "radius": int(rng.integers(*ranges["defocus_radius"])),
        }
    elif blur_type == "gaussian":
        k_size = int(rng.integers(*ranges["gauss_size"]))
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        return {
            "size": k_size,
            "sigma": float(rng.uniform(*ranges["gauss_sigma"])),
        }
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
