"""Export trained model to ONNX and TFLite formats.

Usage:
    python export.py --model output/blur_detector_best.pth --output-dir ../models

Exports:
    - blur_detector.onnx          (ONNX format, dynamic batch)
    - blur_detector.tflite         (TFLite format, float32)
    - blur_detector_quant.tflite   (TFLite format, INT8 quantized)
"""

import argparse
import os
import sys

import numpy as np
import torch

from model.config import Config
from model.network import build_model


def export_onnx(model, output_path: str, image_size: int = 224, opset: int = 13):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"✓ Exported ONNX model: {output_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model verified successfully")
    except ImportError:
        print(f"  (Install 'onnx' to verify the exported model)")


def export_tflite_float(model, output_path: str, image_size: int = 224):
    """Export model to TFLite float32 format via ONNX intermediate."""
    try:
        import onnx
        from onnxruntime.transformers import optimizer
    except ImportError:
        pass

    # Method 1: Direct PyTorch → TFLite via torch.export (PyTorch 2.0+)
    try:
        import torch.export
        exported = torch.export.export(model, (torch.randn(1, 3, image_size, image_size),))

        # Use executorch or ai_edge_torch if available
        try:
            import ai_edge_torch
            edge_model = ai_edge_torch.convert(exported, (torch.randn(1, 3, image_size, image_size),))
            edge_model.export(output_path)
            print(f"✓ Exported TFLite model (ai_edge_torch): {output_path}")
            return
        except ImportError:
            pass
    except Exception as e:
        print(f"  torch.export failed: {e}")

    # Method 2: PyTorch → ONNX → TFLite via tf2onnx
    onnx_path = output_path.replace(".tflite", ".onnx")
    if not os.path.exists(onnx_path):
        export_onnx(model, onnx_path, image_size)

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # ONNX → TF
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_model_dir = output_path.replace(".tflite", "_tf")
        tf_rep.export_graph(tf_model_dir)

        # TF → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"✓ Exported TFLite model: {output_path}")

        # Cleanup
        import shutil
        if os.path.exists(tf_model_dir):
            shutil.rmtree(tf_model_dir)

    except ImportError as e:
        print(f"  Cannot export TFLite (missing dependency: {e})")
        print(f"  Install: pip install onnx-tf tensorflow")
        print(f"  Alternative: use onnx2tf tool")
        # Fallback: save ONNX only and provide manual instructions
        print(f"\n  Manual conversion options:")
        print(f"    1. pip install onnx2tf && onnx2tf -i {onnx_path} -o {output_path}")
        print(f"    2. pip install ai_edge_torch && use ai_edge_torch.convert()")


def export_tflite_quantized(model, output_path: str, image_size: int = 224,
                             calibration_data: np.ndarray = None):
    """Export INT8 quantized TFLite model.

    Uses Post-Training Quantization (PTQ) with calibration dataset.
    Converts via ONNX → TF SavedModel → quantized TFLite pipeline.
    """
    try:
        import tensorflow as tf

        # Step 1: Export to ONNX if not already done
        onnx_path = output_path.replace("_quant.tflite", ".onnx")
        if not os.path.exists(onnx_path):
            print(f"  Exporting ONNX intermediate for quantization...")
            export_onnx(model, onnx_path, image_size)

        if not os.path.exists(onnx_path):
            print(f"  Cannot quantize: ONNX export failed")
            return

        # Step 2: ONNX → TF SavedModel
        try:
            from onnx_tf.backend import prepare
            import onnx

            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_model_dir = output_path.replace(".tflite", "_tf")
            tf_rep.export_graph(tf_model_dir)
        except ImportError:
            print(f"  Cannot quantize: missing onnx-tf package")
            print(f"  Install: pip install onnx-tf")
            return
        except Exception as e:
            print(f"  ONNX→TF conversion failed: {e}")
            return

        # Step 3: TF SavedModel → INT8 quantized TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if calibration_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(calibration_data))):
                        yield [calibration_data[i:i+1].astype(np.float32)]
                converter.representative_dataset = representative_dataset

            quantized_model = converter.convert()

            with open(output_path, "wb") as f:
                f.write(quantized_model)

            quant_size = len(quantized_model) / (1024 * 1024)
            print(f"✓ Exported quantized TFLite model: {output_path}")
            print(f"  INT8 size: {quant_size:.2f} MB")

        finally:
            # Cleanup TF saved model directory
            import shutil
            if os.path.exists(tf_model_dir):
                shutil.rmtree(tf_model_dir)

    except Exception as e:
        print(f"  Quantization failed: {e}")
        print(f"  Try: pip install onnx-tf tensorflow")


def generate_calibration_data(image_size: int = 224, num_samples: int = 100):
    """Generate random calibration data for quantization.

    In production, use real images from the training set.
    """
    return np.random.randn(num_samples, 3, image_size, image_size).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Export blur detection model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", default="../models", help="Output directory")
    parser.add_argument("--backbone", default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--onnx-opset", type=int, default=13)
    parser.add_argument("--quantize", action="store_true", help="Also export INT8 quantized model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cpu")

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = build_model(
        num_classes=4,
        backbone=args.backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, "blur_detector.onnx")
    export_onnx(model, onnx_path, args.image_size, args.onnx_opset)

    # Export TFLite float32
    tflite_path = os.path.join(args.output_dir, "blur_detector.tflite")
    export_tflite_float(model, tflite_path, args.image_size)

    # Export quantized if requested
    if args.quantize:
        quant_path = os.path.join(args.output_dir, "blur_detector_quant.tflite")
        calib_data = generate_calibration_data(args.image_size)
        export_tflite_quantized(model, quant_path, args.image_size, calib_data)

    print("\n=== Export Complete ===")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
