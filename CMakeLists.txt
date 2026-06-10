cmake_minimum_required(VERSION 3.16)
project(BlurDetection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────
option(BLUR_DETECTION_BUILD_TESTS    "Build unit tests"        ON)
option(BLUR_DETECTION_BUILD_APP      "Build CLI application"   ON)

# ── Dependencies ─────────────────────────────────────────
include(FetchContent)

# stb (header-only image I/O)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_SOURCE_DIR}/third_party/stb)

# ── ONNX Runtime Backend ─────────────────────────────────
if(DEFINED ONNXRUNTIME_ROOT)
    if(APPLE)
        set(ORT_LIB_NAME "libonnxruntime.dylib")
    elseif(UNIX)
        set(ORT_LIB_NAME "libonnxruntime.so")
    else()
        set(ORT_LIB_NAME "onnxruntime.dll")
    endif()

    set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
    set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB_DIR}/${ORT_LIB_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    )
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ORT_LIB_DIR}/onnxruntime.lib"
        )
    endif()
    message(STATUS "ONNX Runtime: using pre-built library at ${ONNXRUNTIME_ROOT}")
else()
    find_library(ORT_LIB onnxruntime
        PATHS /usr/local/lib /usr/lib /opt/homebrew/lib
    )
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /usr/local/include/onnxruntime /usr/include /opt/homebrew/include/onnxruntime
    )
    if(ORT_LIB AND ORT_INCLUDE_DIR)
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        )
        message(STATUS "ONNX Runtime: found system library at ${ORT_LIB}")
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install via 'brew install onnxruntime' or set -DONNXRUNTIME_ROOT=<path>")
    endif()
endif()

# ── Google Test ──────────────────────────────────────────
if(BLUR_DETECTION_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Library Sources ──────────────────────────────────────
set(BLUR_LIB_SOURCES
    src/image_processor.cpp
    src/blur_detector.cpp
    src/inference_engine.cpp
    src/onnx_engine.cpp
)
set(BLUR_LIB_HEADERS
    include/blur_detection/inference_engine.h
    include/blur_detection/image_processor.h
    include/blur_detection/blur_detector.h
    include/blur_detection/onnx_engine.h
)

# ── Library Target ───────────────────────────────────────
add_library(blur_detection_lib STATIC ${BLUR_LIB_SOURCES} ${BLUR_LIB_HEADERS})
target_include_directories(blur_detection_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(blur_detection_lib PUBLIC stb_image onnxruntime)

# ── CLI Application ──────────────────────────────────────
if(BLUR_DETECTION_BUILD_APP)
    add_executable(blur_detection apps/main.cpp)
    target_link_libraries(blur_detection PRIVATE blur_detection_lib)
endif()

# ── Tests ────────────────────────────────────────────────
if(BLUR_DETECTION_BUILD_TESTS)
    add_executable(blur_detection_tests
        tests/test_blur_detector.cpp
        tests/test_cli_format.cpp
    )
    target_link_libraries(blur_detection_tests PRIVATE
        blur_detection_lib
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(blur_detection_tests)
endif()

# ── Summary ──────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== BlurDetection Build Configuration ===")
message(STATUS "  ONNX Runtime:    ON")
message(STATUS "  Build tests:     ${BLUR_DETECTION_BUILD_TESTS}")
message(STATUS "  Build app:       ${BLUR_DETECTION_BUILD_APP}")
message(STATUS "==========================================")
message(STATUS "")
