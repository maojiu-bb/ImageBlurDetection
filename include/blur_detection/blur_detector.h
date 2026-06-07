#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "blur_detection/inference_engine.h"

namespace blur {

/// Blur classification categories.
enum class BlurClass : int {
    kSharp       = 0,
    kMotionBlur  = 1,
    kDefocusBlur = 2,
    kGaussianBlur = 3,
    kCount       = 4
};

/// Convert BlurClass to human-readable string.
const char* blurClassName(BlurClass cls);

/// Result of a blur detection inference.
struct BlurResult {
    BlurClass predicted_class = BlurClass::kSharp;
    float confidence = 0.0f;
    std::array<float, static_cast<int>(BlurClass::kCount)> probabilities = {};

    /// Get class name as string.
    const char* className() const { return blurClassName(predicted_class); }
};

/// Top-level blur detector. Wraps an inference engine and image preprocessor.
class BlurDetector {
public:
    /// Construct with a model file and backend type.
    /// @param model_path  Path to .tflite or .onnx model file.
    /// @param backend     "tflite" or "onnx".
    /// @param input_size  Model input spatial size (default 224).
    BlurDetector(const std::string& model_path,
                 const std::string& backend = "tflite",
                 int input_size = 224);

    /// Detect blur in an image file.
    BlurResult detect(const std::string& image_path);

    /// Detect blur from raw pixel data (HWC, uint8, RGB).
    BlurResult detect(const uint8_t* data, int width, int height, int channels);

    /// Check if the detector is ready (model loaded).
    bool isReady() const;

private:
    BlurResult runInference(const std::vector<float>& input_tensor);

    std::unique_ptr<InferenceEngine> engine_;
    int input_size_;
};

}  // namespace blur
