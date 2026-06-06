#pragma once

#ifdef BLUR_HAS_ONNX

#include "blur_detection/inference_engine.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

namespace blur {

/// ONNX Runtime inference backend.
class OnnxEngine : public InferenceEngine {
public:
    OnnxEngine();
    ~OnnxEngine() override;

    bool loadModel(const std::string& model_path) override;
    std::vector<float> predict(const float* input_data,
                               const std::vector<int64_t>& input_shape) override;
    std::vector<int64_t> getInputShape() const override;
    std::vector<int64_t> getOutputShape() const override;
    std::string backendName() const override { return "onnx"; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace blur

#endif  // BLUR_HAS_ONNX
