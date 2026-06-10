"""Training configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────
    dataset_root: str = "data/datasets"
    image_size: int = 224
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["defocus_blur", "gaussian_blur", "motion_blur", "sharp"]
    )

    # ── Training ─────────────────────────────────────────
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10

    # ── Model ────────────────────────────────────────────
    backbone: str = "mobilenet_v3_small"  # "mobilenet_v3_small" or "mobilenet_v3_large"
    pretrained: bool = True
    dropout: float = 0.2

    # ── Data Augmentation ────────────────────────────────
    random_horizontal_flip: float = 0.5
    random_rotation: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # ── Export ───────────────────────────────────────────
    export_onnx: bool = True
    export_tflite: bool = True
    quantize_int8: bool = True
    onnx_opset: int = 13

    # ── Misc ─────────────────────────────────────────────
    num_workers: int = 4
    seed: int = 42
    output_dir: str = "output"
    model_save_name: str = "blur_detector"
