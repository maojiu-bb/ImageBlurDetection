#include "blur_detection/blur_detector.h"
#include "blur_detection/image_processor.h"
#include "blur_detection/inference_engine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace blur {

const char* blurClassName(BlurClass cls) {
    switch (cls) {
        case BlurClass::kDefocusBlur:  return "defocus_blur";
        case BlurClass::kGaussianBlur: return "gaussian_blur";
        case BlurClass::kMotionBlur:   return "motion_blur";
        case BlurClass::kSharp:        return "sharp";
        default:                       return "unknown";
    }
}

BlurDetector::BlurDetector(const std::string& model_path,
                           const std::string& backend,
                           int input_size)
    : input_size_(input_size) {
    engine_ = InferenceEngine::create(backend);
    if (!engine_) {
        throw std::runtime_error("BlurDetector: Unknown backend '" + backend + "'");
    }

    if (!engine_->loadModel(model_path)) {
        throw std::runtime_error("BlurDetector: Failed to load model from " + model_path);
    }
}

BlurResult BlurDetector::detect(const std::string& image_path) {
    auto tensor = ImageProcessor::preprocess(image_path, input_size_);
    if (tensor.empty()) {
        std::cerr << "BlurDetector: Failed to preprocess image: " << image_path << std::endl;
        return {};
    }
    return runInference(tensor);
}

BlurResult BlurDetector::detect(const uint8_t* data, int width, int height, int channels) {
    auto tensor = ImageProcessor::preprocess(data, width, height, channels, input_size_);
    if (tensor.empty()) {
        std::cerr << "BlurDetector: Failed to preprocess raw image data" << std::endl;
        return {};
    }
    return runInference(tensor);
}

bool BlurDetector::isReady() const {
    return engine_ != nullptr;
}

BlurResult BlurDetector::runInference(const std::vector<float>& input_tensor) {
    // Input shape: {1, 3, input_size_, input_size_}
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};

    std::vector<float> output = engine_->predict(input_tensor.data(), input_shape);

    BlurResult result;

    if (output.size() < static_cast<size_t>(BlurClass::kCount)) {
        std::cerr << "BlurDetector: Unexpected output size: " << output.size() << std::endl;
        return result;
    }

    // Softmax
    float max_val = *std::max_element(output.begin(),
                                       output.begin() + static_cast<int>(BlurClass::kCount));
    float sum = 0.0f;
    for (int i = 0; i < static_cast<int>(BlurClass::kCount); ++i) {
        result.probabilities[i] = std::exp(output[i] - max_val);
        sum += result.probabilities[i];
    }
    for (int i = 0; i < static_cast<int>(BlurClass::kCount); ++i) {
        result.probabilities[i] /= sum;
    }

    // Find argmax
    int best_idx = 0;
    float best_prob = result.probabilities[0];
    for (int i = 1; i < static_cast<int>(BlurClass::kCount); ++i) {
        if (result.probabilities[i] > best_prob) {
            best_prob = result.probabilities[i];
            best_idx = i;
        }
    }

    result.predicted_class = static_cast<BlurClass>(best_idx);
    result.confidence = best_prob;

    return result;
}

}  // namespace blur
