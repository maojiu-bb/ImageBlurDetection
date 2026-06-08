# Image Blur Detection — 端侧模型

端侧照片模糊检测系统：多类别分类（清晰 / 运动模糊 / 失焦模糊 / 高斯模糊），使用 ONNX Runtime 进行推理。

## 模型信息

| 项目 | 说明 |
|------|------|
| **Backbone** | MobileNetV3-Large (~5.4M 参数) |
| **输入** | 224×224 RGB 图像，ImageNet 归一化 |
| **输出** | 4 类 softmax 概率 |
| **准确率** | 验证集 98.84%，测试集 97.80% (TTA) |
| **模型格式** | ONNX (12MB) |

### 分类类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | defocus_blur | 失焦模糊（对焦不准） |
| 1 | gaussian_blur | 高斯模糊（整体柔和） |
| 2 | motion_blur | 运动模糊（相机/物体移动） |
| 3 | sharp | 清晰图像 |

> ⚠️ 类别顺序按 ImageFolder 字母排序，与模型输出索引对应。

---

## 快速开始

### 1. 训练模型

```bash
cd training

# 安装依赖
pip install -r requirements.txt

# 准备数据集（合成模糊图像，每张源图生成 15 个变体）
python data/prepare_dataset.py --source data/sharp_images --output data/datasets --variants 15

# 训练
python train.py --data-dir data/datasets --epochs 100

# 评估
python evaluate.py --model output/blur_detector_best.pth --data-dir data/datasets

# 导出 ONNX 模型
python export.py --model output/blur_detector_best.pth --output-dir ../models
```

### 2. 构建 C++ 推理引擎

```bash
# 安装 ONNX Runtime
brew install onnxruntime

# 构建
cmake -B build -DONNXRUNTIME_ROOT=/opt/homebrew/opt/onnxruntime
cmake --build build

# 运行
./build/blur_detection --model models/blur_detector.onnx --backend onnx --input photo.jpg
```

### 3. CLI 用法

```bash
# 单图检测
./build/blur_detection --model models/blur_detector.onnx --backend onnx --input photo.jpg

# 批量检测
./build/blur_detection --model models/blur_detector.onnx --backend onnx --input ./photos/

# JSON 输出
./build/blur_detection --model models/blur_detector.onnx --backend onnx --input photo.jpg --json
```

输出示例：
```
photo.jpg: motion_blur (confidence: 96.16%) [0.013, 0.013, 0.962, 0.013]
```

---

## 跨平台集成指南

### iOS / macOS (Swift)

使用 [onnxruntime-swift](https://github.com/microsoft/onnxruntime) 官方库：

```swift
import onnxruntime_objc

class BlurDetector {
    private var session: ORTSession?
    private let classes = ["defocus_blur", "gaussian_blur", "motion_blur", "sharp"]
    
    init(modelPath: String) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        options.setIntraOpNumThreads(4)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }
    
    func detect(image: CGImage) throws -> (className: String, confidence: Float, probabilities: [Float]) {
        // 1. 预处理: resize 到 224x224, 转换为 CHW float32, ImageNet 归一化
        let inputData = preprocessImage(image)
        
        // 2. 创建输入 tensor
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: [1, 3, 224, 224]
        )
        
        // 3. 推理
        let outputs = try session?.run(
            withInputs: ["input": inputTensor],
            outputNames: ["output"]
        )
        
        // 4. 获取结果
        let outputTensor = outputs?["output"]
        let outputData = try outputTensor?.tensorData() as Data?
        let probabilities = outputData?.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        } ?? []
        
        // 5. Softmax + argmax
        let maxVal = probabilities.max() ?? 0
        let expValues = probabilities.map { exp($0 - maxVal) }
        let sum = expValues.reduce(0, +)
        let probs = expValues.map { $0 / sum }
        
        let maxIndex = probs.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        
        return (classes[maxIndex], probs[maxIndex], probs)
    }
    
    private func preprocessImage(_ image: CGImage) -> Data {
        let width = 224
        let height = 224
        
        // 1. 绘制到 224x224 RGB 画布
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: nil,
                  width: width, height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: width * 4,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                      CGImageAlphaInfo.noneSkipFirst.rawValue
              ) else {
            return Data()
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelData = context.data else { return Data() }
        let pixels = pixelData.bindMemory(to: UInt8.self, capacity: width * height * 4)
        
        // 2. HWC -> CHW + ImageNet 归一化
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        let channelSize = width * height
        
        var floatData = [Float](repeating: 0, count: 3 * channelSize)
        for i in 0..<channelSize {
            let offset = i * 4  // BGRA 布局
            for c in 0..<3 {
                let pixel = Float(pixels[offset + c]) / 255.0
                floatData[c * channelSize + i] = (pixel - mean[c]) / std[c]
            }
        }
        
        return Data(bytes: floatData, count: floatData.count * MemoryLayout<Float>.stride)
    }
}

// 使用
let detector = try BlurDetector(modelPath: "blur_detector.onnx")
let result = try detector.detect(image: uiImage.cgImage!)
print("检测结果: \(result.className), 置信度: \(result.confidence)")
```

**Podfile:**
```ruby
pod 'onnxruntime-objc', '~> 1.16'
```
Swift 中 `import onnxruntime_objc`，Objective-C 中 `#import <onnxruntime_objc/onnxruntime_objc.h>`。

**Swift Package Manager:**
```swift
dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime.git", from: "1.16.0")
]
```

---

### iOS / macOS (Objective-C)

```objectivec
#import <onnxruntime_objc/onnxruntime_objc.h>

@interface BlurDetector : NSObject
- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;
- (NSDictionary *)detectWithImage:(CGImageRef)image error:(NSError **)error;
@end

@implementation BlurDetector {
    ORTSession *_session;
    NSArray<NSString *> *_classes;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
    self = [super init];
    if (self) {
        _classes = @[@"defocus_blur", @"gaussian_blur", @"motion_blur", @"sharp"];
        
        ORTEnv *env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:error];
        ORTSessionOptions *options = [[ORTSessionOptions alloc] init];
        [options setIntraOpNumThreads:4 error:error];
        
        _session = [[ORTSession alloc] initWithEnv:env modelPath:modelPath sessionOptions:options error:error];
    }
    return self;
}

- (NSDictionary *)detectWithImage:(CGImageRef)image error:(NSError **)error {
    // 1. 预处理图像
    NSData *inputData = [self preprocessImage:image];
    
    // 2. 创建输入 tensor
    ORTValue *inputTensor = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithData:inputData]
                                                       elementType:ORTTensorElementDataTypeFloat
                                                             shape:@[@1, @3, @224, @224]
                                                             error:error];
    
    // 3. 推理
    NSDictionary<NSString *, ORTValue *> *outputs = [_session runWithInputs:@{@"input": inputTensor}
                                                                outputNames:@[@"output"]
                                                                      error:error];
    
    // 4. 解析输出（模型输出已是 softmax 概率）
    ORTValue *outputTensor = outputs[@"output"];
    NSData *outputData = [outputTensor tensorDataWithError:error];
    const float *probs = (const float *)outputData.bytes;
    
    // 5. 找到最大概率
    int maxIdx = 0;
    float maxProb = probs[0];
    for (int i = 1; i < 4; i++) {
        if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIdx = i;
        }
    }
    
    return @{
        @"className": _classes[maxIdx],
        @"confidence": @(maxProb),
        @"probabilities": @[@(probs[0]), @(probs[1]), @(probs[2]), @(probs[3])]
    };
}

- (NSData *)preprocessImage:(CGImageRef)image {
    const int width = 224;
    const int height = 224;
    
    // 1. 绘制到 224x224 RGB 画布
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef context = CGBitmapContextCreate(
        NULL, width, height, 8, width * 4,
        colorSpace,
        kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst
    );
    CGColorSpaceRelease(colorSpace);
    
    if (!context) return [NSData data];
    
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    
    const uint8_t *pixels = CGBitmapContextGetData(context);
    if (!pixels) {
        CGContextRelease(context);
        return [NSData data];
    }
    
    // 2. HWC -> CHW + ImageNet 归一化
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};
    const int channelSize = width * height;
    
    NSMutableData *outputData = [NSMutableData dataWithLength:3 * channelSize * sizeof(float)];
    float *floatBuffer = (float *)outputData.mutableBytes;
    
    for (int i = 0; i < channelSize; i++) {
        int offset = i * 4;  // BGRA 布局
        for (int c = 0; c < 3; c++) {
            float pixel = pixels[offset + c] / 255.0f;
            floatBuffer[c * channelSize + i] = (pixel - mean[c]) / std[c];
        }
    }
    
    CGContextRelease(context);
    return outputData;
}

@end
```

---

### Android (Kotlin)

使用 [onnxruntime-android](https://github.com/microsoft/onnxruntime) 官方库：

**build.gradle:**
```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

**BlurDetector.kt:**
```kotlin
import ai.onnxruntime.*
import android.graphics.Bitmap
import java.nio.FloatBuffer

class BlurDetector(context: Context, modelPath: String) {
    
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val classes = arrayOf("defocus_blur", "gaussian_blur", "motion_blur", "sharp")
    
    init {
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(4)
        val modelBytes = context.assets.open(modelPath).readBytes()
        session = env.createSession(modelBytes, options)
    }
    
    fun detect(bitmap: Bitmap): Pair<String, Float> {
        // 1. 预处理
        val inputTensor = preprocessImage(bitmap)
        
        // 2. 推理
        val results = session.run(mapOf("input" to inputTensor))
        
        // 3. 获取输出
        val output = (results[0].value as Array<FloatArray>)[0]
        
        // 4. Softmax
        val maxVal = output.max()
        val expValues = output.map { Math.exp((it - maxVal).toDouble()).toFloat() }
        val sum = expValues.sum()
        val probs = expValues.map { it / sum }
        
        // 5. Argmax
        val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        
        return Pair(classes[maxIndex], probs[maxIndex])
    }
    
    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        // HWC -> CHW, 归一化
        for (c in 0 until 3) {
            for (y in 0 until 224) {
                for (x in 0 until 224) {
                    val pixel = resized.getPixel(x, y)
                    val value = when (c) {
                        0 -> (android.graphics.Color.red(pixel) / 255.0f - mean[c]) / std[c]
                        1 -> (android.graphics.Color.green(pixel) / 255.0f - mean[c]) / std[c]
                        else -> (android.graphics.Color.blue(pixel) / 255.0f - mean[c]) / std[c]
                    }
                    floatBuffer.put(value)
                }
            }
        }
        
        floatBuffer.rewind()
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 224, 224))
    }
    
    fun close() {
        session.close()
        env.close()
    }
}

// 使用
val detector = BlurDetector(context, "blur_detector.onnx")
val (className, confidence) = detector.detect(bitmap)
println("检测结果: $className, 置信度: $confidence")
detector.close()
```

**将模型放入 `assets/` 目录：**
```
app/src/main/assets/blur_detector.onnx
```

---

### Flutter (Dart)

使用 [onnxruntime](https://pub.dev/packages/onnxruntime) 插件：

**pubspec.yaml:**
```yaml
dependencies:
  onnxruntime: ^1.4.0
  image: ^4.1.0
```

**blur_detector.dart:**
```dart
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

class BlurDetector {
  late OrtSession _session;
  final List<String> classes = [
    'defocus_blur', 'gaussian_blur', 'motion_blur', 'sharp'
  ];

  Future<void> init(String modelPath) async {
    final sessionOptions = OrtSessionOptions();
    sessionOptions.setIntraOpNumThreads(4);
    
    final session = OrtSession.fromFile(
      await File(modelPath).readAsBytes(),
      sessionOptions,
    );
    _session = session;
  }

  Future<Map<String, dynamic>> detect(Uint8List imageBytes) async {
    // 1. 解码图像
    final image = img.decodeImage(imageBytes)!;
    
    // 2. 预处理
    final inputData = _preprocessImage(image);
    
    // 3. 创建输入 tensor
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, 224, 224],
    );
    
    // 4. 推理
    final inputs = {'input': inputTensor};
    final outputs = await _session.runAsync(OrtRunOptions(), inputs);
    
    // 5. 获取结果
    final outputTensor = outputs[0];
    final outputData = outputTensor.value as List<double>;
    
    // 6. Softmax
    final maxVal = outputData.reduce((a, b) => a > b ? a : b);
    final expValues = outputData.map((v) => exp(v - maxVal)).toList();
    final sum = expValues.reduce((a, b) => a + b);
    final probs = expValues.map((v) => v / sum).toList();
    
    // 7. Argmax
    final maxIndex = probs.indexOf(probs.reduce((a, b) => a > b ? a : b));
    
    inputTensor.release();
    outputTensor.release();
    
    return {
      'className': classes[maxIndex],
      'confidence': probs[maxIndex],
      'probabilities': probs,
    };
  }

  Float32List _preprocessImage(img.Image image) {
    // Resize 到 224x224
    final resized = img.copyResize(image, width: 224, height: 224);
    
    final inputData = Float32List(1 * 3 * 224 * 224);
    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];
    
    int index = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          final value = switch (c) {
            0 => (pixel.r / 255.0 - mean[c]) / std[c],
            1 => (pixel.g / 255.0 - mean[c]) / std[c],
            _ => (pixel.b / 255.0 - mean[c]) / std[c],
          };
          inputData[index++] = value;
        }
      }
    }
    
    return inputData;
  }

  void dispose() {
    _session.release();
  }
}

// 使用
final detector = BlurDetector();
await detector.init('assets/blur_detector.onnx');
final result = await detector.detect(imageBytes);
print('检测结果: ${result['className']}, 置信度: ${result['confidence']}');
detector.dispose();
```

---

### Python

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

class BlurDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.classes = ['defocus_blur', 'gaussian_blur', 'motion_blur', 'sharp']
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def detect(self, image_path: str) -> dict:
        # 加载并预处理
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        
        # HWC -> CHW, 添加 batch 维度
        input_data = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]
        
        # 推理
        outputs = self.session.run(None, {'input': input_data})
        probs = self._softmax(outputs[0][0])
        
        # 结果
        max_idx = np.argmax(probs)
        return {
            'className': self.classes[max_idx],
            'confidence': float(probs[max_idx]),
            'probabilities': probs.tolist()
        }
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

# 使用
detector = BlurDetector('models/blur_detector.onnx')
result = detector.detect('photo.jpg')
print(f"检测结果: {result['className']}, 置信度: {result['confidence']:.2%}")
```

---

### Web (JavaScript)

使用 [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)：

```javascript
import * as ort from 'onnxruntime-web';

class BlurDetector {
    constructor() {
        this.classes = ['defocus_blur', 'gaussian_blur', 'motion_blur', 'sharp'];
        this.mean = [0.485, 0.456, 0.406];
        this.std = [0.229, 0.224, 0.225];
    }
    
    async init(modelPath) {
        this.session = await ort.InferenceSession.create(modelPath);
    }
    
    async detect(imageElement) {
        // 1. 预处理
        const inputData = this.preprocessImage(imageElement);
        
        // 2. 创建 tensor
        const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
        
        // 3. 推理
        const results = await this.session.run({ input: inputTensor });
        const output = results.output.data;
        
        // 4. Softmax
        const probs = this.softmax(Array.from(output));
        
        // 5. Argmax
        const maxIdx = probs.indexOf(Math.max(...probs));
        
        return {
            className: this.classes[maxIdx],
            confidence: probs[maxIdx],
            probabilities: probs
        };
    }
    
    preprocessImage(imageElement) {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0, 224, 224);
        
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const data = imageData.data;
        
        const inputData = new Float32Array(3 * 224 * 224);
        for (let c = 0; c < 3; c++) {
            for (let y = 0; y < 224; y++) {
                for (let x = 0; x < 224; x++) {
                    const pixelIdx = (y * 224 + x) * 4;
                    const value = (data[pixelIdx + c] / 255.0 - this.mean[c]) / this.std[c];
                    inputData[c * 224 * 224 + y * 224 + x] = value;
                }
            }
        }
        
        return inputData;
    }
    
    softmax(arr) {
        const maxVal = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - maxVal));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }
}

// 使用
const detector = new BlurDetector();
await detector.init('blur_detector.onnx');
const result = await detector.detect(document.getElementById('photo'));
console.log(`检测结果: ${result.className}, 置信度: ${(result.confidence * 100).toFixed(2)}%`);
```

---

### React Native

使用 [react-native-onnx](https://github.com/nicko170/react-native-onnx) 或通过 Native Module 调用原生 ONNX Runtime。

---

## 预处理规范

所有平台的预处理必须一致：

1. **Resize**: 将图像缩放到 224×224
2. **色彩空间**: 转换为 RGB（去除 Alpha 通道）
3. **数据布局**: HWC → CHW（通道优先）
4. **归一化**: `(pixel / 255.0 - mean) / std`
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`
5. **数据类型**: float32
6. **输入形状**: `[1, 3, 224, 224]`

---

## 项目结构

```
├── CMakeLists.txt              # C++ 构建配置
├── models/
│   └── blur_detector.onnx      # 训练好的模型
├── training/                   # Python 训练管线
│   ├── data/
│   │   ├── blur_kernels.py     # 模糊核生成
│   │   └── prepare_dataset.py  # 数据集准备
│   ├── model/
│   │   ├── config.py           # 训练配置
│   │   └── network.py          # MobileNetV3 模型
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估脚本
│   └── export.py               # 导出 ONNX
├── include/blur_detection/     # C++ 头文件
├── src/                        # C++ 源文件
├── apps/main.cpp               # CLI 工具
└── tests/                      # 单元测试
```

## 依赖

### Python（训练）
- PyTorch >= 2.0
- torchvision >= 0.15
- onnx >= 1.14

### C++（推理）
- ONNX Runtime (`brew install onnxruntime`)
- CMake >= 3.16
- C++17 编译器

## License

MIT
