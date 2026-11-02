#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>

#include <android/log.h>
#define LOG_TAG "DINOv3_Native"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
constexpr int PATCH_SIZE = 16;
constexpr float MEAN_R = 0.485f * 255.0f;
constexpr float MEAN_G = 0.456f * 255.0f;
constexpr float MEAN_B = 0.406f * 255.0f;
constexpr float STD_R = 0.229f;
constexpr float STD_G = 0.224f;
constexpr float STD_B = 0.225f;

Ort::Env* g_env = nullptr;
Ort::Session* g_session = nullptr;

// Helper to preprocess RGB cv::Mat into normalized NCHW ONNX blob
cv::Mat preprocess_image(const cv::Mat& rgb, int input_size, int& out_w_patches, int& out_h_patches) {
    int h_patches = input_size / PATCH_SIZE;
    int w_patches = (rgb.cols * input_size) / (rgb.rows * PATCH_SIZE);
    if (w_patches % 2 != 0) w_patches -= 1;

    out_w_patches = w_patches;
    out_h_patches = h_patches;

    int new_w = w_patches * PATCH_SIZE;
    int new_h = h_patches * PATCH_SIZE;

    // Resize image (bicubic interpolation)
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // Convert HWC uint8 -> NCHW float32 and subtract ImageNet mean
    cv::Mat blob = cv::dnn::blobFromImage(
        resized,
        1.0 / 255.0,
        cv::Size(new_w, new_h),
        cv::Scalar(MEAN_R, MEAN_G, MEAN_B),
        false, // swapRB (already RGB)
        false  // crop
    );

    // Fast vectorized channel-specific std division using OpenCV matrix views
    cv::Mat ch0(new_h, new_w, CV_32F, blob.ptr<float>(0, 0));
    ch0 /= STD_R;

    cv::Mat ch1(new_h, new_w, CV_32F, blob.ptr<float>(0, 1));
    ch1 /= STD_G;

    cv::Mat ch2(new_h, new_w, CV_32F, blob.ptr<float>(0, 2));
    ch2 /= STD_B;

    return blob;
}

// Runs inference on an NCHW blob and returns the output tensor
std::vector<Ort::Value> run_inference(const cv::Mat& blob, int new_w, int new_h) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = g_session->GetInputNameAllocated(0, allocator);
    auto output_name_alloc = g_session->GetOutputNameAllocated(0, allocator);

    const char* input_names[] = { input_name_alloc.get() };
    const char* output_names[] = { output_name_alloc.get() };

    std::vector<int64_t> input_shape = {1, 3, new_h, new_w};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(blob.data),
        blob.total(),
        input_shape.data(),
        input_shape.size()
    );

    return g_session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
}
} // namespace

extern "C" {

__attribute__((visibility("default"))) __attribute__((used))
void init_session(const char* model_path) {
    try {
        if (g_session != nullptr) {
            delete g_session;
            g_session = nullptr;
        }
        if (g_env != nullptr) {
            delete g_env;
            g_env = nullptr;
        }

        g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DINOv3_Native");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);

        #if defined(__ANDROID__)
        // If NNAPI fails or is unsupported on this device/chipset, log and continue with CPU
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*)session_options, 0);
        if (status != nullptr) {
            const auto& api = Ort::GetApi();
            LOGE("NNAPI Provider registration failed: %s", api.GetErrorMessage(status));
            api.ReleaseStatus(status);
        }
        #endif

        g_session = new Ort::Session(*g_env, model_path, session_options);
        LOGE("Session successfully created for model: %s", model_path);
    } catch (const Ort::Exception& e) {
        LOGE("ONNX Runtime Exception in init_session: %s", e.what());
    } catch (const std::exception& e) {
        LOGE("Standard Exception in init_session: %s", e.what());
    }
}

__attribute__((visibility("default"))) __attribute__((used))
float* create_prototype(uint8_t* rgba_data, int length, int input_size, int* out_feature_dim) {
    if (!g_session || !rgba_data || length <= 0) {
        *out_feature_dim = 0;
        return nullptr;
    }

    // Decode image buffer from memory (PNG with transparency)
    cv::Mat buffer(1, length, CV_8UC1, rgba_data);
    cv::Mat decoded = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (decoded.empty() || decoded.channels() != 4) {
        *out_feature_dim = 0;
        return nullptr;
    }

    // Separate RGB and Alpha channels
    cv::Mat rgb, mask;
    std::vector<cv::Mat> channels;
    cv::split(decoded, channels);
    mask = channels[3]; // Alpha channel (0 = transparent, 255 = opaque)
    cv::cvtColor(decoded, rgb, cv::COLOR_BGRA2RGB);

    // Preprocess RGB image
    int w_patches = 0, h_patches = 0;
    cv::Mat blob = preprocess_image(rgb, input_size, w_patches, h_patches);
    int new_w = w_patches * PATCH_SIZE;
    int new_h = h_patches * PATCH_SIZE;

    // Downscale alpha mask to patch grid using nearest-neighbor interpolation
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, cv::Size(w_patches, h_patches), 0, 0, cv::INTER_NEAREST);

    // Run ONNX inference
    auto output_tensors = run_inference(blob, new_w, new_h);
    const float* all_features = output_tensors.front().GetTensorData<float>();
    size_t total_elements = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    int num_patches = w_patches * h_patches;
    int feature_dim = static_cast<int>(total_elements / num_patches);

    // Aggregate features from foreground patches (Alpha > 127)
    std::vector<float> prototype(feature_dim, 0.0f);
    int fg_count = 0;

    for (int i = 0; i < num_patches; ++i) {
        if (resized_mask.data[i] > 127) {
            const float* patch_feat = all_features + (i * feature_dim);
            for (int d = 0; d < feature_dim; ++d) {
                prototype[d] += patch_feat[d];
            }
            fg_count++;
        }
    }

    if (fg_count == 0) {
        *out_feature_dim = 0;
        return nullptr;
    }

    for (int d = 0; d < feature_dim; ++d) {
        prototype[d] /= static_cast<float>(fg_count);
    }

    // Allocate memory returning to Dart (Dart will free this via free_pointer)
    float* result = new float[feature_dim];
    std::memcpy(result, prototype.data(), feature_dim * sizeof(float));
    *out_feature_dim = feature_dim;

    return result;
}

__attribute__((visibility("default"))) __attribute__((used))
float* run_segmentation(
    uint8_t* yuv_data,
    int width,
    int height,
    float* prototype,
    int feature_dim,
    double threshold,
    int input_size,
    bool largest_only,
    int* out_w,
    int* out_h
) {
    if (!g_session || !yuv_data || !prototype) {
        *out_w = 0;
        *out_h = 0;
        return nullptr;
    }

    // Convert YUV420 to RGB and rotate 90 degrees clockwise (camera portrait orientation)
    cv::Mat yuv(height * 3 / 2, width, CV_8UC1, yuv_data);
    cv::Mat rgb;
    cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
    cv::Mat rotated;
    cv::rotate(rgb, rotated, cv::ROTATE_90_CLOCKWISE);

    // Preprocess rotated image
    int w_patches = 0, h_patches = 0;
    cv::Mat blob = preprocess_image(rotated, input_size, w_patches, h_patches);
    *out_w = w_patches;
    *out_h = h_patches;
    int num_patches = w_patches * h_patches;

    // Run ONNX Inference
    auto output_tensors = run_inference(blob, w_patches * PATCH_SIZE, h_patches * PATCH_SIZE);
    const float* test_features = output_tensors.front().GetTensorData<float>();

    // Compute Cosine Similarity between reference prototype and each patch
    float* similarity_scores = new float[num_patches];

    // Precalculate prototype magnitude once
    float proto_mag = 0.0f;
    for (int d = 0; d < feature_dim; ++d) {
        proto_mag += prototype[d] * prototype[d];
    }
    proto_mag = std::sqrt(proto_mag);

    for (int i = 0; i < num_patches; ++i) {
        const float* patch_feat = test_features + (i * feature_dim);
        float dot = 0.0f;
        float patch_mag = 0.0f;

        for (int d = 0; d < feature_dim; ++d) {
            dot += patch_feat[d] * prototype[d];
            patch_mag += patch_feat[d] * patch_feat[d];
        }
        patch_mag = std::sqrt(patch_mag);

        similarity_scores[i] = (proto_mag > 0.0f && patch_mag > 0.0f)
            ? (dot / (proto_mag * patch_mag))
            : 0.0f;
    }

    // Largest Area Filtering (Connected Components)
    if (largest_only) {
        cv::Mat mask_mat(h_patches, w_patches, CV_8UC1);
        for (int i = 0; i < num_patches; ++i) {
            mask_mat.data[i] = (similarity_scores[i] > threshold) ? 255 : 0;
        }

        cv::Mat labels, stats, centroids;
        int n_labels = cv::connectedComponentsWithStats(
            mask_mat,
            labels,
            stats,
            centroids,
            8,
            CV_32S,
            cv::CCL_DEFAULT
        );

        if (stats.rows > 1) {
            int max_area = 0;
            int largest_component_label = 0;

            // Start from 1 to skip background (label 0)
            for (int i = 1; i < stats.rows; ++i) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > max_area) {
                    max_area = area;
                    largest_component_label = i;
                }
            }

            if (largest_component_label != 0) {
                const int* labels_data = reinterpret_cast<const int*>(labels.data);
                for (int i = 0; i < num_patches; ++i) {
                    if (labels_data[i] != largest_component_label) {
                        similarity_scores[i] = 0.0f;
                    }
                }
            }
        }
    }

    return similarity_scores;
}

__attribute__((visibility("default"))) __attribute__((used))
void free_pointer(void* ptr) {
    if (ptr != nullptr) {
        delete[] reinterpret_cast<float*>(ptr);
    }
}

__attribute__((visibility("default"))) __attribute__((used))
bool save_yuv_for_gemma(
    uint8_t* yuv_data,
    int width,
    int height,
    int max_dim,
    const char* out_jpeg_path
) {
    if (!yuv_data || !out_jpeg_path) return false;

    cv::Mat yuv(height * 3 / 2, width, CV_8UC1, yuv_data);
    cv::Mat bgr, rotated;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
    cv::rotate(bgr, rotated, cv::ROTATE_90_CLOCKWISE);

    float scale = static_cast<float>(max_dim) / std::max(rotated.cols, rotated.rows);
    if (scale < 1.0f) {
        cv::resize(rotated, rotated, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    return cv::imwrite(out_jpeg_path, rotated);
}

__attribute__((visibility("default"))) __attribute__((used))
float* create_prototype_from_box(
    uint8_t* yuv_data,
    int width,
    int height,
    float ymin,
    float xmin,
    float ymax,
    float xmax,
    int input_size,
    int* out_feature_dim
) {
    if (!g_session || !yuv_data) {
        *out_feature_dim = 0;
        return nullptr;
    }

    // Convert YUV420 to RGB and rotate 90 degrees clockwise to match portrait camera
    cv::Mat yuv(height * 3 / 2, width, CV_8UC1, yuv_data);
    cv::Mat rgb, rotated;
    cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
    cv::rotate(rgb, rotated, cv::ROTATE_90_CLOCKWISE);

    // Preprocess and run DINOv3 ONNX inference
    int w_patches = 0, h_patches = 0;
    cv::Mat blob = preprocess_image(rotated, input_size, w_patches, h_patches);
    int new_w = w_patches * PATCH_SIZE;
    int new_h = h_patches * PATCH_SIZE;

    auto output_tensors = run_inference(blob, new_w, new_h);
    const float* all_features = output_tensors.front().GetTensorData<float>();
    size_t total_elements = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    int num_patches = w_patches * h_patches;
    int feature_dim = static_cast<int>(total_elements / num_patches);

    // Map normalized [0.0, 1.0] box coordinates to DINOv3 patch grid
    int x0 = std::clamp(static_cast<int>(std::floor(xmin * w_patches)), 0, w_patches - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(xmax * w_patches)) - 1, x0, w_patches - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(ymin * h_patches)), 0, h_patches - 1);
    int y1 = std::clamp(static_cast<int>(std::ceil(ymax * h_patches)) - 1, y0, h_patches - 1);

    // Step 1 of Refinement: Average the inner 36% core to form a pure foreground seed
    int box_w = x1 - x0 + 1;
    int box_h = y1 - y0 + 1;
    int margin_x = static_cast<int>(box_w * 0.32f);
    int margin_y = static_cast<int>(box_h * 0.32f);

    int core_x0 = std::min(x0 + margin_x, x1);
    int core_x1 = std::max(x1 - margin_x, core_x0);
    int core_y0 = std::min(y0 + margin_y, y1);
    int core_y1 = std::max(y1 - margin_y, core_y0);

    std::vector<float> seed_proto(feature_dim, 0.0f);
    int seed_count = 0;

    for (int y = core_y0; y <= core_y1; ++y) {
        for (int x = core_x0; x <= core_x1; ++x) {
            const float* patch_feat = all_features + ((y * w_patches + x) * feature_dim);
            for (int d = 0; d < feature_dim; ++d) {
                seed_proto[d] += patch_feat[d];
            }
            seed_count++;
        }
    }

    float seed_mag = 0.0f;
    for (int d = 0; d < feature_dim; ++d) {
        seed_proto[d] /= static_cast<float>(std::max(1, seed_count));
        seed_mag += seed_proto[d] * seed_proto[d];
    }
    seed_mag = std::sqrt(seed_mag);

    // Step 2 of Refinement: Filter all patches inside the bounding box against the seed
    std::vector<float> final_proto(feature_dim, 0.0f);
    int fg_count = 0;
    constexpr float SEED_SIM_THRESHOLD = 0.60f;

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float* patch_feat = all_features + ((y * w_patches + x) * feature_dim);
            float dot = 0.0f;
            float patch_mag = 0.0f;

            for (int d = 0; d < feature_dim; ++d) {
                dot += patch_feat[d] * seed_proto[d];
                patch_mag += patch_feat[d] * patch_feat[d];
            }
            patch_mag = std::sqrt(patch_mag);

            float sim = (seed_mag > 0.0f && patch_mag > 0.0f)
                ? (dot / (seed_mag * patch_mag))
                : 0.0f;

            if (sim >= SEED_SIM_THRESHOLD) {
                for (int d = 0; d < feature_dim; ++d) {
                    final_proto[d] += patch_feat[d];
                }
                fg_count++;
            }
        }
    }

    // Fallback to seed_proto if threshold filtered everything out (e.g. tiny 1-patch box)
    if (fg_count == 0) {
        final_proto = seed_proto;
        fg_count = 1;
    } else {
        for (int d = 0; d < feature_dim; ++d) {
            final_proto[d] /= static_cast<float>(fg_count);
        }
    }

    float* result = new float[feature_dim];
    std::memcpy(result, final_proto.data(), feature_dim * sizeof(float));
    *out_feature_dim = feature_dim;

    return result;
}

} // extern "C"
