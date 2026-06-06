#ifdef BLUR_HAS_TFLITE

#include "blur_detection/tflite_engine.h"

#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace blur {

TFLiteEngine::TFLiteEngine() = default;

TFLiteEngine::~TFLiteEngine() = default;

bool TFLiteEngine::loadModel(const std::string& model_path) {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) {
        std::cerr << "TFLite: Failed to load model from " << model_path << std::endl;
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);

    if (!interpreter_) {
        std::cerr << "TFLite: Failed to build interpreter" << std::endl;
        return false;
    }

    // Use XNNPack delegate for CPU acceleration
    interpreter_->SetNumThreads(4);

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "TFLite: Failed to allocate tensors" << std::endl;
        return false;
    }

    return true;
}

std::vector<float> TFLiteEngine::predict(const float* input_data,
                                          const std::vector<int64_t>& input_shape) {
    if (!interpreter_) {
        throw std::runtime_error("TFLite: Model not loaded");
    }

    // Copy input data to the interpreter's input tensor
    float* input_ptr = interpreter_->typed_input_tensor<float>(0);
    if (!input_ptr) {
        throw std::runtime_error("TFLite: Failed to get input tensor pointer");
    }

    size_t num_elements = 1;
    for (auto dim : input_shape) {
        num_elements *= static_cast<size_t>(dim);
    }
    std::memcpy(input_ptr, input_data, num_elements * sizeof(float));

    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("TFLite: Inference failed");
    }

    // Read output
    const TfLiteTensor* output_tensor = interpreter_->output_tensor(0);
    const float* output_ptr = output_tensor->data.f;
    size_t output_size = 1;
    for (int i = 0; i < output_tensor->dims->size; ++i) {
        output_size *= output_tensor->dims->data[i];
    }

    return std::vector<float>(output_ptr, output_ptr + output_size);
}

std::vector<int64_t> TFLiteEngine::getInputShape() const {
    if (!interpreter_) return {};

    const TfLiteTensor* tensor = interpreter_->input_tensor(0);
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor->dims->size; ++i) {
        shape.push_back(tensor->dims->data[i]);
    }
    return shape;
}

std::vector<int64_t> TFLiteEngine::getOutputShape() const {
    if (!interpreter_) return {};

    const TfLiteTensor* tensor = interpreter_->output_tensor(0);
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor->dims->size; ++i) {
        shape.push_back(tensor->dims->data[i]);
    }
    return shape;
}

}  // namespace blur

#endif  // BLUR_HAS_TFLITE
