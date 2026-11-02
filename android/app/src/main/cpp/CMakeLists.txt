cmake_minimum_required(VERSION 3.10.0)
project(dinov3_native)

include(FetchContent)

# Fetch ONNX Runtime directly from Maven Central
FetchContent_Declare(
  onnxruntime
  URL https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.30.0/onnxruntime-android-1.30.0.aar
)
FetchContent_MakeAvailable(onnxruntime)

# Fetch OpenCV Android SDK (Extract only, do not build samples)
FetchContent_Declare(
  opencv
  URL https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-android-sdk.zip
)
FetchContent_GetProperties(opencv)
if(NOT opencv_POPULATED)
  FetchContent_Populate(opencv)
endif()

# Safely locate the JNI folder depending on how your CMake version extracted the ZIP
if(EXISTS "${opencv_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni/OpenCVConfig.cmake")
    set(OpenCV_DIR "${opencv_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni")
elseif(EXISTS "${opencv_SOURCE_DIR}/sdk/native/jni/OpenCVConfig.cmake")
    set(OpenCV_DIR "${opencv_SOURCE_DIR}/sdk/native/jni")
else()
    message(FATAL_ERROR "Could not find OpenCVConfig.cmake in the downloaded OpenCV SDK at ${opencv_SOURCE_DIR}")
endif()

find_package(OpenCV REQUIRED)

# Build your native C++ code
add_library(dinov3_native SHARED native_segmentation.cpp)

# Link the extracted AAR contents and OpenCV
# The ONNX AAR places C headers in /headers and native libraries in /jni
target_include_directories(dinov3_native PRIVATE ${onnxruntime_SOURCE_DIR}/headers)

target_link_libraries(dinov3_native PRIVATE 
    ${onnxruntime_SOURCE_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so
    ${OpenCV_LIBS}
    android 
    log
)