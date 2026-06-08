#include "blur_detection/onnx_engine.h"

#include <iostream>
#include <stdexcept>

namespace blur {

OnnxEngine::OnnxEngine() = default;

OnnxEngine::~OnnxEngine() = default;

bool OnnxEngine::loadModel(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BlurDetection");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Get input info
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes == 0) {
            std::cerr << "ONNX: Model has no inputs" << std::endl;
            return false;
        }

        auto input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
        input_name_ = input_name_alloc.get();

        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();

        // Get output info
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes == 0) {
            std::cerr << "ONNX: Model has no outputs" << std::endl;
            return false;
        }

        auto output_name_alloc = session_->GetOutputNameAllocated(0, allocator_);
        output_name_ = output_name_alloc.get();

        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX: Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> OnnxEngine::predict(const float* input_data,
                                        const std::vector<int64_t>& input_shape) {
    if (!session_) {
        throw std::runtime_error("ONNX: Model not loaded");
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor
        size_t num_elements = 1;
        for (auto dim : input_shape) {
            num_elements *= static_cast<size_t>(dim);
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data), num_elements,
            input_shape.data(), input_shape.size());

        // Run inference
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // Extract output
        const float* output_data = output_tensors[0].GetTensorData<float>();
        size_t output_size = 1;
        for (auto dim : output_shape_) {
            if (dim > 0) output_size *= static_cast<size_t>(dim);
        }

        return std::vector<float>(output_data, output_data + output_size);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX: Inference failed: ") + e.what());
    }
}

std::vector<int64_t> OnnxEngine::getInputShape() const {
    return input_shape_;
}

std::vector<int64_t> OnnxEngine::getOutputShape() const {
    return output_shape_;
}

}  // namespace blur
