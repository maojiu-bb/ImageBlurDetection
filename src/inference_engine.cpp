#include "blur_detection/inference_engine.h"
#include "blur_detection/onnx_engine.h"

#include <algorithm>
#include <stdexcept>

namespace blur {

std::unique_ptr<InferenceEngine> InferenceEngine::create(const std::string& backend) {
    std::string lower_backend = backend;
    std::transform(lower_backend.begin(), lower_backend.end(),
                   lower_backend.begin(), ::tolower);

    if (lower_backend == "onnx") {
        return std::make_unique<OnnxEngine>();
    }

    return nullptr;
}

}  // namespace blur
