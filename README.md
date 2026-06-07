# Image Blur Detection — On-Device Model

端侧照片模糊检测系统：多类别分类（清晰 / 运动模糊 / 失焦模糊 / 高斯模糊），使用 C++ 推理，支持 TFLite 和 ONNX Runtime 双后端。

## 架构

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  训练 (PyTorch) │ ──→ │ 导出 (.tflite │ ──→ │ C++ 推理引擎       │
│  MobileNetV3  │     │   / .onnx)   │     │ TFLite / ONNX RT  │
└─────────────┘     └──────────────┘     └───────────────────┘
```

## 快速开始

### 1. 训练模型

```bash
cd training

# 安装依赖
pip install -r requirements.txt

# 准备数据集（合成模糊图像，每张源图生成 10 个模糊变体）
python data/prepare_dataset.py --source data/sharp_images --output data/datasets --variants 10

# 训练
python train.py --data-dir data/datasets --epochs 50

# 评估
python evaluate.py --model output/blur_detector_best.pth --data-dir data/datasets

# 导出模型
python export.py --model output/blur_detector_best.pth --output-dir ../models
```

### 2. 构建 C++ 推理引擎

```bash
# 安装推理框架依赖（以 macOS Homebrew 为例）
brew install tensorflow-lite onnxruntime

# 构建
cmake -B build \
  -DTFLITE_ROOT=/path/to/tflite \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build

# 运行
./build/blur_detection \
  --model models/blur_detector.tflite \
  --backend tflite \
  --input path/to/image.jpg
```

### 3. CLI 用法

```bash
# 单图检测
./blur_detection --model models/blur_detector.onnx --backend onnx --input photo.jpg

# 批量检测
./blur_detection --model models/blur_detector.tflite --input ./photos/

# JSON 输出
./blur_detection --model models/blur_detector.tflite --input photo.jpg --json
```

输出示例：
```
photo.jpg: motion_blur (confidence: 92.3%) [0.02, 0.92, 0.04, 0.02]
```

## 分类类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | sharp | 清晰图像 |
| 1 | motion_blur | 运动模糊（相机/物体移动） |
| 2 | defocus_blur | 失焦模糊（对焦不准） |
| 3 | gaussian_blur | 高斯模糊（整体柔和） |

## 项目结构

```
├── CMakeLists.txt              # C++ 构建配置
├── training/                   # Python 训练管线
│   ├── data/
│   │   ├── blur_kernels.py     # 模糊核生成
│   │   └── prepare_dataset.py  # 数据集准备
│   ├── model/
│   │   ├── config.py           # 训练配置
│   │   └── network.py          # MobileNetV3 模型
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估脚本
│   └── export.py               # 导出 TFLite/ONNX
├── include/blur_detection/     # C++ 头文件
├── src/                        # C++ 源文件
├── apps/main.cpp               # CLI 工具
├── models/                     # 导出的模型文件
└── tests/                      # 单元测试
```

## 模型说明

- **Backbone**: MobileNetV3-Small（~2.5M 参数，适合端侧部署）
- **输入**: 224×224 RGB 图像，ImageNet 归一化
- **输出**: 4 类 softmax 概率
- **量化**: 支持 INT8 PTQ 量化（体积减小 ~4x）

## 依赖

### Python（训练）
- PyTorch >= 2.0
- torchvision >= 0.15
- onnx >= 1.14

### C++（推理）
- TFLite C library 或 ONNX Runtime
- CMake >= 3.16
- C++17 编译器

## License

MIT
