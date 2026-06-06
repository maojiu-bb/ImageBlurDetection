#pragma once

#ifdef BLUR_HAS_TFLITE

#include "blur_detection/inference_engine.h"

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include <memory>

namespace blur {

/// TensorFlow Lite inference backend.
class TFLiteEngine : public InferenceEngine {
public:
    TFLiteEngine();
    ~TFLiteEngine() override;

    bool loadModel(const std::string& model_path) override;
    std::vector<float> predict(const float* input_data,
                               const std::vector<int64_t>& input_shape) override;
    std::vector<int64_t> getInputShape() const override;
    std::vector<int64_t> getOutputShape() const override;
    std::string backendName() const override { return "tflite"; }

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace blur

#endif  // BLUR_HAS_TFLITE
