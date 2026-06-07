#include "blur_detection/blur_detector.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  --model <path>      Path to model file (.tflite or .onnx)\n"
              << "  --backend <name>    Inference backend: tflite | onnx (default: tflite)\n"
              << "  --input <path>      Input image or directory\n"
              << "  --json              Output results as JSON\n"
              << "  --help              Show this help\n"
              << "\nBlur Classes:\n"
              << "  0 = defocus_blur, 1 = gaussian_blur, 2 = motion_blur, 3 = sharp\n";
}

void printResult(const std::string& image_path, const blur::BlurResult& result, bool json) {
    if (json) {
        std::cout << "{"
                  << "\"file\":\"" << image_path << "\","
                  << "\"class\":\"" << result.className() << "\","
                  << "\"class_id\":" << static_cast<int>(result.predicted_class) << ","
                  << "\"confidence\":" << result.confidence << ","
                  << "\"probabilities\":{"
                  << "\"sharp\":" << result.probabilities[0] << ","
                  << "\"motion_blur\":" << result.probabilities[1] << ","
                  << "\"defocus_blur\":" << result.probabilities[2] << ","
                  << "\"gaussian_blur\":" << result.probabilities[3]
                  << "}}\n";
    } else {
        std::cout << image_path << ": "
                  << result.className()
                  << " (confidence: " << (result.confidence * 100.0f) << "%)"
                  << " [" << result.probabilities[0]
                  << ", " << result.probabilities[1]
                  << ", " << result.probabilities[2]
                  << ", " << result.probabilities[3] << "]"
                  << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string backend = "tflite";
    std::string input_path;
    bool json_output = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty() || input_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    // Initialize detector
    std::unique_ptr<blur::BlurDetector> detector;
    try {
        detector = std::make_unique<blur::BlurDetector>(model_path, backend);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Collect image files
    std::vector<std::string> image_files;
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".webp"
    };

    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                image_files.push_back(entry.path().string());
            }
        }
        std::sort(image_files.begin(), image_files.end());
    } else {
        image_files.push_back(input_path);
    }

    if (image_files.empty()) {
        std::cerr << "No image files found in: " << input_path << std::endl;
        return 1;
    }

    const size_t total = image_files.size();
    if (!json_output && total > 1) {
        std::cerr << "Processing " << total << " images..." << std::endl;
    }

    if (json_output) {
        std::cout << "[\n";
    }

    size_t processed = 0;
    size_t failed = 0;
    for (size_t i = 0; i < image_files.size(); ++i) {
        blur::BlurResult result = detector->detect(image_files[i]);

        // Check if inference succeeded (default result means failure)
        if (result.confidence == 0.0f && result.predicted_class == blur::BlurClass::kSharp) {
            ++failed;
        }

        printResult(image_files[i], result, json_output);

        if (json_output && i + 1 < image_files.size()) {
            std::cout << ",";
        }

        ++processed;
        // Show progress for batch processing (not in JSON mode)
        if (!json_output && total > 1) {
            std::cerr << "\r  [" << processed << "/" << total << "] processed"
                      << (failed > 0 ? ", " + std::to_string(failed) + " failed" : "")
                      << std::flush;
        }
    }

    if (!json_output && total > 1) {
        std::cerr << std::endl;
    }

    if (json_output) {
        std::cout << "]\n";
    }

    return 0;
}
