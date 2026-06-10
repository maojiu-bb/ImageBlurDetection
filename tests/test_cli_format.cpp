#define main blur_detection_cli_main
#include "../apps/main.cpp"
#undef main

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

namespace {

TEST(CliFormatTest, JsonProbabilitiesMatchBlurClassOrder) {
    blur::BlurResult result;
    result.predicted_class = blur::BlurClass::kMotionBlur;
    result.confidence = 0.3f;
    result.probabilities = {0.1f, 0.2f, 0.3f, 0.4f};

    std::ostringstream output;
    auto* old_buffer = std::cout.rdbuf(output.rdbuf());
    printResult("photo.jpg", result, true);
    std::cout.rdbuf(old_buffer);

    EXPECT_EQ(
        output.str(),
        "{\"file\":\"photo.jpg\",\"class\":\"motion_blur\",\"class_id\":2,"
        "\"confidence\":0.3,\"probabilities\":{\"defocus_blur\":0.1,"
        "\"gaussian_blur\":0.2,\"motion_blur\":0.3,\"sharp\":0.4}}\n"
    );
}

}  // namespace
