"""MobileNetV3-based blur detection model.

Architecture:
    MobileNetV3 backbone → Global Average Pooling → FC classifier head

Supports:
    - mobilenet_v3_small (faster, smaller)
    - mobilenet_v3_large (more accurate)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BlurDetectionModel(nn.Module):
    """Blur classification model based on MobileNetV3.

    Args:
        num_classes: Number of output classes (default: 4).
        backbone: Backbone architecture name.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout probability in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        if backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features  # 576
        elif backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features  # 960
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Replace the original classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)


def build_model(num_classes: int = 4, backbone: str = "mobilenet_v3_small",
                pretrained: bool = True, dropout: float = 0.2) -> BlurDetectionModel:
    """Build a blur detection model.

    Args:
        num_classes: Number of output classes.
        backbone: "mobilenet_v3_small" or "mobilenet_v3_large".
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate in classifier head.

    Returns:
        BlurDetectionModel instance.
    """
    return BlurDetectionModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick model summary
    model = build_model()
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
