#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace blur {

/// Image preprocessing utilities for blur detection inference.
class ImageProcessor {
public:
    /// Default input size for MobileNetV3.
    static constexpr int kDefaultInputSize = 224;

    /// ImageNet normalization constants.
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

    struct ImageData {
        std::vector<uint8_t> pixels;  // Raw RGB pixels (HWC)
        int width  = 0;
        int height = 0;
        int channels = 0;
    };

    /// Load an image from file path. Returns empty ImageData on failure.
    static ImageData loadImage(const std::string& path);

    /// Resize image to target_size x target_size using bilinear interpolation.
    /// Returns resized pixel data in HWC format.
    static ImageData resize(const ImageData& img, int target_size = kDefaultInputSize);

    /// Convert HWC uint8 image to CHW float tensor with ImageNet normalization.
    /// Output shape: {1, 3, height, width}.
    static std::vector<float> toFloatTensor(const ImageData& img);

    /// Full pipeline: load → resize → normalize → tensor.
    /// Returns empty vector on failure.
    static std::vector<float> preprocess(const std::string& image_path,
                                          int input_size = kDefaultInputSize);

    /// Preprocess from raw pixel data (HWC, uint8).
    static std::vector<float> preprocess(const uint8_t* data, int width, int height,
                                          int channels, int input_size = kDefaultInputSize);
};

}  // namespace blur
