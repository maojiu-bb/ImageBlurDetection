#!/bin/bash
# Build TFLite C Library for macOS ARM64
set -e

echo "=== Building TFLite C Library for macOS ARM64 ==="
echo "This will take 10-15 minutes..."
echo ""

brew install cmake git

echo "1/4: Cloning TensorFlow source..."
rm -rf /tmp/tensorflow_src
git clone --depth 1 --branch v2.16.2 https://github.com/tensorflow/tensorflow.git /tmp/tensorflow_src

echo "2/4: Building TFLite C library..."
cd /tmp/tensorflow_src
mkdir -p build && cd build
cmake ../tensorflow/lite/c -DTFLITE_ENABLE_XNNPACK=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build . -j$(sysctl -n hw.ncpu)

echo "3/4: Installing..."
TFLITE_DIR="$HOME/tflite_c"
mkdir -p $TFLITE_DIR/lib $TFLITE_DIR/include
cp libtensorflowlite_c.dylib $TFLITE_DIR/lib/
cp ../tensorflow/lite/c/*.h $TFLITE_DIR/include/

echo "4/4: Building project..."
cd /Users/zhongyu/Documents/Workspace/ImageBlurDetection
cmake -B build -DBLUR_DETECTION_ENABLE_TFLITE=ON -DBLUR_DETECTION_ENABLE_ONNX=ON -DTFLITE_ROOT=$TFLITE_DIR -DONNXRUNTIME_ROOT=/opt/homebrew/opt/onnxruntime
cmake --build build

echo ""
echo "=== Done! ==="
echo "Usage: ./build/blur_detection --model models/blur_detector.tflite --backend tflite --input photo.jpg"
