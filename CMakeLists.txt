cmake_minimum_required(VERSION 3.16)
project(BlurDetection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────
option(BLUR_DETECTION_ENABLE_TFLITE  "Build with TensorFlow Lite backend"  ON)
option(BLUR_DETECTION_ENABLE_ONNX    "Build with ONNX Runtime backend"     ON)
option(BLUR_DETECTION_BUILD_TESTS    "Build unit tests"                     ON)
option(BLUR_DETECTION_BUILD_APP      "Build CLI application"               ON)

# ── Dependencies ─────────────────────────────────────────
include(FetchContent)

# stb (header-only image I/O)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_SOURCE_DIR}/third_party/stb)

# ── TFLite Backend ───────────────────────────────────────
if(BLUR_DETECTION_ENABLE_TFLITE)
    # Users can set TFLITE_ROOT to the extracted TFLite C library directory
    # e.g. cmake -DTFLITE_ROOT=/path/to/tflite_c
    if(DEFINED TFLITE_ROOT)
        set(TFLITE_INCLUDE_DIR "${TFLITE_ROOT}/include")
        set(TFLITE_LIB_DIR "${TFLITE_ROOT}/lib")
        add_library(tensorflowlite_c SHARED IMPORTED)
        set_target_properties(tensorflowlite_c PROPERTIES
            IMPORTED_LOCATION "${TFLITE_LIB_DIR}/libtensorflowlite_c.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_INCLUDE_DIR}"
        )
        # Linux: .so, Windows: .dll
        if(UNIX AND NOT APPLE)
            set_target_properties(tensorflowlite_c PROPERTIES
                IMPORTED_LOCATION "${TFLITE_LIB_DIR}/libtensorflowlite_c.so"
            )
        elseif(WIN32)
            set_target_properties(tensorflowlite_c PROPERTIES
                IMPORTED_LOCATION "${TFLITE_LIB_DIR}/tensorflowlite_c.dll"
                IMPORTED_IMPLIB "${TFLITE_LIB_DIR}/tensorflowlite_c.lib"
            )
        endif()
        message(STATUS "TFLite: using pre-built library at ${TFLITE_ROOT}")
    else()
        # Try to find system-installed TFLite
        find_library(TFLITE_LIB tensorflowlite_c
            PATHS /usr/local/lib /usr/lib /opt/homebrew/lib
        )
        find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h
            PATHS /usr/local/include /usr/include /opt/homebrew/include
        )
        if(TFLITE_LIB AND TFLITE_INCLUDE_DIR)
            add_library(tensorflowlite_c SHARED IMPORTED)
            set_target_properties(tensorflowlite_c PROPERTIES
                IMPORTED_LOCATION "${TFLITE_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_INCLUDE_DIR}"
            )
            message(STATUS "TFLite: found system library at ${TFLITE_LIB}")
        else()
            message(WARNING "TFLite not found. Set -DTFLITE_ROOT=<path> or disable with -DBLUR_DETECTION_ENABLE_TFLITE=OFF")
            set(BLUR_DETECTION_ENABLE_TFLITE OFF)
        endif()
    endif()
endif()

# ── ONNX Runtime Backend ─────────────────────────────────
if(BLUR_DETECTION_ENABLE_ONNX)
    # Users can set ONNXRUNTIME_ROOT to the extracted ONNX Runtime directory
    if(DEFINED ONNXRUNTIME_ROOT)
        if(APPLE)
            set(ORT_LIB_NAME "libonnxruntime.dylib")
            set(ORT_LIB_EXT "dylib")
        elseif(UNIX)
            set(ORT_LIB_NAME "libonnxruntime.so")
            set(ORT_LIB_EXT "so")
        else()
            set(ORT_LIB_NAME "onnxruntime.dll")
            set(ORT_LIB_EXT "dll")
        endif()

        set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
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
            PATHS /usr/local/include /usr/include /opt/homebrew/include
        )
        if(ORT_LIB AND ORT_INCLUDE_DIR)
            add_library(onnxruntime SHARED IMPORTED)
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
            )
            message(STATUS "ONNX Runtime: found system library at ${ORT_LIB}")
        else()
            message(WARNING "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT=<path> or disable with -DBLUR_DETECTION_ENABLE_ONNX=OFF")
            set(BLUR_DETECTION_ENABLE_ONNX OFF)
        endif()
    endif()
endif()

# ── Google Test ──────────────────────────────────────────
if(BLUR_DETECTION_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Compile Definitions ──────────────────────────────────
add_compile_definitions(
    $<$<BOOL:${BLUR_DETECTION_ENABLE_TFLITE}>:BLUR_HAS_TFLITE>
    $<$<BOOL:${BLUR_DETECTION_ENABLE_ONNX}>:BLUR_HAS_ONNX>
)

# ── Library Sources ──────────────────────────────────────
set(BLUR_LIB_SOURCES
    src/image_processor.cpp
    src/blur_detector.cpp
    src/inference_engine.cpp
)
set(BLUR_LIB_HEADERS
    include/blur_detection/inference_engine.h
    include/blur_detection/image_processor.h
    include/blur_detection/blur_detector.h
)

if(BLUR_DETECTION_ENABLE_TFLITE)
    list(APPEND BLUR_LIB_SOURCES src/tflite_engine.cpp)
    list(APPEND BLUR_LIB_HEADERS include/blur_detection/tflite_engine.h)
endif()

if(BLUR_DETECTION_ENABLE_ONNX)
    list(APPEND BLUR_LIB_SOURCES src/onnx_engine.cpp)
    list(APPEND BLUR_LIB_HEADERS include/blur_detection/onnx_engine.h)
endif()

# ── Library Target ───────────────────────────────────────
add_library(blur_detection_lib STATIC ${BLUR_LIB_SOURCES} ${BLUR_LIB_HEADERS})
target_include_directories(blur_detection_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(blur_detection_lib PUBLIC stb_image)

if(BLUR_DETECTION_ENABLE_TFLITE)
    target_link_libraries(blur_detection_lib PUBLIC tensorflowlite_c)
endif()

if(BLUR_DETECTION_ENABLE_ONNX)
    target_link_libraries(blur_detection_lib PUBLIC onnxruntime)
endif()

# ── CLI Application ──────────────────────────────────────
if(BLUR_DETECTION_BUILD_APP)
    add_executable(blur_detection apps/main.cpp)
    target_link_libraries(blur_detection PRIVATE blur_detection_lib)
endif()

# ── Tests ────────────────────────────────────────────────
if(BLUR_DETECTION_BUILD_TESTS)
    add_executable(blur_detection_tests tests/test_blur_detector.cpp)
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
message(STATUS "  TFLite backend:  ${BLUR_DETECTION_ENABLE_TFLITE}")
message(STATUS "  ONNX backend:    ${BLUR_DETECTION_ENABLE_ONNX}")
message(STATUS "  Build tests:     ${BLUR_DETECTION_BUILD_TESTS}")
message(STATUS "  Build app:       ${BLUR_DETECTION_BUILD_APP}")
message(STATUS "==========================================")
message(STATUS "")
