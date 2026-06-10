#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace blur {

/// Abstract interface for neural network inference backends.
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    /// Load a model from the given file path.
    /// Returns true on success.
    virtual bool loadModel(const std::string& model_path) = 0;

    /// Run inference on the given input tensor.
    /// @param input_data  Pointer to float tensor data (CHW layout).
    /// @param input_shape Shape of the input tensor (e.g. {1, 3, 224, 224}).
    /// @returns Output tensor as a flat float vector.
    virtual std::vector<float> predict(const float* input_data,
                                       const std::vector<int64_t>& input_shape) = 0;

    /// Get the expected input shape of the loaded model.
    virtual std::vector<int64_t> getInputShape() const = 0;

    /// Get the output shape of the loaded model.
    virtual std::vector<int64_t> getOutputShape() const = 0;

    /// Get the backend name (e.g. "onnx").
    virtual std::string backendName() const = 0;

    /// Factory: create an engine by backend name ("onnx").
    /// Returns nullptr if the requested backend is not available.
    static std::unique_ptr<InferenceEngine> create(const std::string& backend);
};

}  // namespace blur
