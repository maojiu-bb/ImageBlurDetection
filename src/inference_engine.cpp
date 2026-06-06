#include "blur_detection/inference_engine.h"

#ifdef BLUR_HAS_TFLITE
#include "blur_detection/tflite_engine.h"
#endif

#ifdef BLUR_HAS_ONNX
#include "blur_detection/onnx_engine.h"
#endif

#include <algorithm>
#include <stdexcept>

namespace blur {

std::unique_ptr<InferenceEngine> InferenceEngine::create(const std::string& backend) {
    std::string lower_backend = backend;
    std::transform(lower_backend.begin(), lower_backend.end(),
                   lower_backend.begin(), ::tolower);

#ifdef BLUR_HAS_TFLITE
    if (lower_backend == "tflite") {
        return std::make_unique<TFLiteEngine>();
    }
#endif

#ifdef BLUR_HAS_ONNX
    if (lower_backend == "onnx") {
        return std::make_unique<OnnxEngine>();
    }
#endif

    return nullptr;
}

}  // namespace blur
