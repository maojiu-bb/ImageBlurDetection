#include "blur_detection/blur_detector.h"
#include "blur_detection/image_processor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// ── ImageProcessor Tests ────────────────────────────────

TEST(ImageProcessorTest, ToFloatTensorShape) {
    // Create a small 2x2 RGB image
    blur::ImageProcessor::ImageData img;
    img.width = 2;
    img.height = 2;
    img.channels = 3;
    img.pixels = {
        255, 0, 0,    // Red
        0, 255, 0,    // Green
        0, 0, 255,    // Blue
        128, 128, 128 // Gray
    };

    auto tensor = blur::ImageProcessor::toFloatTensor(img);

    // Expected shape: 1 * 3 * 2 * 2 = 12
    ASSERT_EQ(tensor.size(), 12u);
}

TEST(ImageProcessorTest, ToFloatTensorNormalization) {
    blur::ImageProcessor::ImageData img;
    img.width = 1;
    img.height = 1;
    img.channels = 3;
    // Pure white pixel: 255, 255, 255
    img.pixels = {255, 255, 255};

    auto tensor = blur::ImageProcessor::toFloatTensor(img);

    // Each channel: (1.0 - mean) / std
    for (int ch = 0; ch < 3; ++ch) {
        float expected = (1.0f - blur::ImageProcessor::kMean[ch]) / blur::ImageProcessor::kStd[ch];
        EXPECT_NEAR(tensor[ch], expected, 1e-5f) << "Channel " << ch;
    }
}

TEST(ImageProcessorTest, ToFloatTensorBlackPixel) {
    blur::ImageProcessor::ImageData img;
    img.width = 1;
    img.height = 1;
    img.channels = 3;
    img.pixels = {0, 0, 0};

    auto tensor = blur::ImageProcessor::toFloatTensor(img);

    // Each channel: (0.0 - mean) / std
    for (int ch = 0; ch < 3; ++ch) {
        float expected = (0.0f - blur::ImageProcessor::kMean[ch]) / blur::ImageProcessor::kStd[ch];
        EXPECT_NEAR(tensor[ch], expected, 1e-5f) << "Channel " << ch;
    }
}

TEST(ImageProcessorTest, ToFloatTensorEmptyInput) {
    blur::ImageProcessor::ImageData img;
    auto tensor = blur::ImageProcessor::toFloatTensor(img);
    EXPECT_TRUE(tensor.empty());
}

TEST(ImageProcessorTest, ToFloatTensorCHWLayout) {
    // Verify CHW layout: all R values first, then G, then B
    blur::ImageProcessor::ImageData img;
    img.width = 2;
    img.height = 1;
    img.channels = 3;
    // Pixel 0: R=100, G=0, B=0
    // Pixel 1: R=0, G=200, B=0
    img.pixels = {100, 0, 0, 0, 200, 0};

    auto tensor = blur::ImageProcessor::toFloatTensor(img);

    // CHW layout: [R0, R1, G0, G1, B0, B1]
    // R channel (index 0,1): pixel 0 has R=100, pixel 1 has R=0
    float r0 = (100.0f / 255.0f - blur::ImageProcessor::kMean[0]) / blur::ImageProcessor::kStd[0];
    float r1 = (0.0f / 255.0f - blur::ImageProcessor::kMean[0]) / blur::ImageProcessor::kStd[0];
    EXPECT_NEAR(tensor[0], r0, 1e-5f);
    EXPECT_NEAR(tensor[1], r1, 1e-5f);
}

TEST(ImageProcessorTest, PreprocessInvalidPath) {
    auto result = blur::ImageProcessor::preprocess("/nonexistent/image.jpg");
    EXPECT_TRUE(result.empty());
}

TEST(ImageProcessorTest, PreprocessNullData) {
    auto result = blur::ImageProcessor::preprocess(nullptr, 0, 0, 0);
    EXPECT_TRUE(result.empty());
}

TEST(ImageProcessorTest, PreprocessZeroDimensions) {
    uint8_t data[3] = {255, 0, 0};
    auto result = blur::ImageProcessor::preprocess(data, 0, 1, 3);
    EXPECT_TRUE(result.empty());

    result = blur::ImageProcessor::preprocess(data, 1, 0, 3);
    EXPECT_TRUE(result.empty());
}

TEST(ImageProcessorTest, PreprocessUnsupportedChannels) {
    uint8_t data[8] = {255, 0, 128, 64, 255, 0, 128, 64};
    auto result = blur::ImageProcessor::preprocess(data, 1, 1, 2);  // 2 channels unsupported
    EXPECT_TRUE(result.empty());
}

TEST(ImageProcessorTest, PreprocessRGBData) {
    // 1x1 RGB image
    uint8_t data[3] = {128, 64, 32};
    auto result = blur::ImageProcessor::preprocess(data, 1, 1, 3);
    ASSERT_EQ(result.size(), 1u * 3 * blur::ImageProcessor::kDefaultInputSize * blur::ImageProcessor::kDefaultInputSize);
}

TEST(ImageProcessorTest, PreprocessRGBAData) {
    // 1x1 RGBA image (should be converted to RGB)
    uint8_t data[4] = {128, 64, 32, 255};
    auto result = blur::ImageProcessor::preprocess(data, 1, 1, 4);
    ASSERT_EQ(result.size(), 1u * 3 * blur::ImageProcessor::kDefaultInputSize * blur::ImageProcessor::kDefaultInputSize);
}

TEST(ImageProcessorTest, PreprocessGrayscaleData) {
    // 1x1 grayscale image (should be converted to RGB)
    uint8_t data[1] = {128};
    auto result = blur::ImageProcessor::preprocess(data, 1, 1, 1);
    ASSERT_EQ(result.size(), 1u * 3 * blur::ImageProcessor::kDefaultInputSize * blur::ImageProcessor::kDefaultInputSize);
}

TEST(ImageProcessorTest, ResizeInvalidInput) {
    blur::ImageProcessor::ImageData img;
    auto resized = blur::ImageProcessor::resize(img);
    EXPECT_TRUE(resized.pixels.empty());
}

// ── BlurClass Tests ─────────────────────────────────────

TEST(BlurClassTest, ClassNames) {
    EXPECT_STREQ(blur::blurClassName(blur::BlurClass::kDefocusBlur), "defocus_blur");
    EXPECT_STREQ(blur::blurClassName(blur::BlurClass::kGaussianBlur), "gaussian_blur");
    EXPECT_STREQ(blur::blurClassName(blur::BlurClass::kMotionBlur), "motion_blur");
    EXPECT_STREQ(blur::blurClassName(blur::BlurClass::kSharp), "sharp");
}

TEST(BlurClassTest, ClassEnumValues) {
    EXPECT_EQ(static_cast<int>(blur::BlurClass::kDefocusBlur), 0);
    EXPECT_EQ(static_cast<int>(blur::BlurClass::kGaussianBlur), 1);
    EXPECT_EQ(static_cast<int>(blur::BlurClass::kMotionBlur), 2);
    EXPECT_EQ(static_cast<int>(blur::BlurClass::kSharp), 3);
    EXPECT_EQ(static_cast<int>(blur::BlurClass::kCount), 4);
}

TEST(BlurClassTest, UnknownClassValue) {
    // Test with an out-of-range enum value
    auto name = blur::blurClassName(static_cast<blur::BlurClass>(99));
    EXPECT_STREQ(name, "unknown");
}

// ── BlurResult Tests ────────────────────────────────────

TEST(BlurResultTest, DefaultValues) {
    blur::BlurResult result;
    EXPECT_EQ(result.predicted_class, blur::BlurClass::kSharp);
    EXPECT_FLOAT_EQ(result.confidence, 0.0f);
    for (float p : result.probabilities) {
        EXPECT_FLOAT_EQ(p, 0.0f);
    }
}

TEST(BlurResultTest, ClassNameMethod) {
    blur::BlurResult result;
    result.predicted_class = blur::BlurClass::kMotionBlur;
    EXPECT_STREQ(result.className(), "motion_blur");
}

// ── InferenceEngine Factory Tests ───────────────────────

TEST(InferenceEngineTest, CreateUnknownBackend) {
    auto engine = blur::InferenceEngine::create("unknown_backend");
    EXPECT_EQ(engine, nullptr);
}

TEST(InferenceEngineTest, CreateCaseInsensitive) {
    // The factory should handle case-insensitive backend names
    auto engine = blur::InferenceEngine::create("UNKNOWN");
    EXPECT_EQ(engine, nullptr);
}

#ifdef BLUR_HAS_TFLITE
TEST(InferenceEngineTest, CreateTFLite) {
    auto engine = blur::InferenceEngine::create("tflite");
    EXPECT_NE(engine, nullptr);
    EXPECT_EQ(engine->backendName(), "tflite");
}

TEST(InferenceEngineTest, CreateTFLiteCaseInsensitive) {
    auto engine = blur::InferenceEngine::create("TFLite");
    EXPECT_NE(engine, nullptr);
    EXPECT_EQ(engine->backendName(), "tflite");
}
#endif

#ifdef BLUR_HAS_ONNX
TEST(InferenceEngineTest, CreateOnnx) {
    auto engine = blur::InferenceEngine::create("onnx");
    EXPECT_NE(engine, nullptr);
    EXPECT_EQ(engine->backendName(), "onnx");
}

TEST(InferenceEngineTest, CreateOnnxCaseInsensitive) {
    auto engine = blur::InferenceEngine::create("ONNX");
    EXPECT_NE(engine, nullptr);
    EXPECT_EQ(engine->backendName(), "onnx");
}
#endif

// ── BlurDetector Construction Tests ─────────────────────

TEST(BlurDetectorTest, InvalidModelPath) {
    EXPECT_THROW(
        blur::BlurDetector("/nonexistent/model.tflite", "tflite"),
        std::runtime_error
    );
}

TEST(BlurDetectorTest, UnknownBackend) {
    EXPECT_THROW(
        blur::BlurDetector("dummy.tflite", "unknown"),
        std::runtime_error
    );
}

// ── ImageNet Normalization Constants Tests ───────────────

TEST(ImageProcessorTest, NormalizationConstants) {
    // Verify ImageNet normalization constants are correct
    EXPECT_NEAR(blur::ImageProcessor::kMean[0], 0.485f, 1e-6f);
    EXPECT_NEAR(blur::ImageProcessor::kMean[1], 0.456f, 1e-6f);
    EXPECT_NEAR(blur::ImageProcessor::kMean[2], 0.406f, 1e-6f);
    EXPECT_NEAR(blur::ImageProcessor::kStd[0], 0.229f, 1e-6f);
    EXPECT_NEAR(blur::ImageProcessor::kStd[1], 0.224f, 1e-6f);
    EXPECT_NEAR(blur::ImageProcessor::kStd[2], 0.225f, 1e-6f);
}

TEST(ImageProcessorTest, DefaultInputSize) {
    EXPECT_EQ(blur::ImageProcessor::kDefaultInputSize, 224);
}

}  // namespace
