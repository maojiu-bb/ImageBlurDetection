#include "blur_detection/image_processor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace blur {

ImageProcessor::ImageData ImageProcessor::loadImage(const std::string& path) {
    ImageData img;
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);  // Force RGB
    if (!data) {
        std::cerr << "ImageProcessor: Failed to load image: " << path << std::endl;
        return {};
    }

    img.width = w;
    img.height = h;
    img.channels = 3;
    img.pixels.assign(data, data + w * h * 3);
    stbi_image_free(data);

    return img;
}

ImageProcessor::ImageData ImageProcessor::resize(const ImageData& img, int target_size) {
    if (img.pixels.empty() || img.width <= 0 || img.height <= 0) {
        return {};
    }

    ImageData resized;
    resized.width = target_size;
    resized.height = target_size;
    resized.channels = 3;
    resized.pixels.resize(target_size * target_size * 3);

    unsigned char* result = stbir_resize_uint8_linear(
        img.pixels.data(), img.width, img.height, 0,
        resized.pixels.data(), target_size, target_size, 0,
        STBIR_RGB
    );

    if (!result) {
        std::cerr << "ImageProcessor: Resize failed" << std::endl;
        return {};
    }

    return resized;
}

std::vector<float> ImageProcessor::toFloatTensor(const ImageData& img) {
    if (img.pixels.empty()) return {};

    const int h = img.height;
    const int w = img.width;
    const int c = 3;

    // Output: CHW layout, shape {1, 3, H, W}
    std::vector<float> tensor(1 * c * h * w);

    for (int ch = 0; ch < c; ++ch) {
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                // HWC source index
                int src_idx = (row * w + col) * c + ch;
                // CHW destination index
                int dst_idx = ch * h * w + row * w + col;

                float val = img.pixels[src_idx] / 255.0f;
                val = (val - kMean[ch]) / kStd[ch];
                tensor[dst_idx] = val;
            }
        }
    }

    return tensor;
}

std::vector<float> ImageProcessor::preprocess(const std::string& image_path, int input_size) {
    ImageData img = loadImage(image_path);
    if (img.pixels.empty()) return {};

    ImageData resized = resize(img, input_size);
    if (resized.pixels.empty()) return {};

    return toFloatTensor(resized);
}

std::vector<float> ImageProcessor::preprocess(const uint8_t* data, int width, int height,
                                               int channels, int input_size) {
    if (!data || width <= 0 || height <= 0) return {};

    ImageData img;
    img.width = width;
    img.height = height;
    img.channels = channels;

    // Convert to RGB if needed
    if (channels == 4) {
        // RGBA → RGB
        img.pixels.resize(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            img.pixels[i * 3 + 0] = data[i * 4 + 0];
            img.pixels[i * 3 + 1] = data[i * 4 + 1];
            img.pixels[i * 3 + 2] = data[i * 4 + 2];
        }
        img.channels = 3;
    } else if (channels == 1) {
        // Grayscale → RGB
        img.pixels.resize(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            img.pixels[i * 3 + 0] = data[i];
            img.pixels[i * 3 + 1] = data[i];
            img.pixels[i * 3 + 2] = data[i];
        }
        img.channels = 3;
    } else if (channels == 3) {
        img.pixels.assign(data, data + width * height * 3);
    } else {
        return {};
    }

    ImageData resized = resize(img, input_size);
    if (resized.pixels.empty()) return {};

    return toFloatTensor(resized);
}

}  // namespace blur
