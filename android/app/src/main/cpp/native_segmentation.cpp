#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include <algorithm>

#include <android/log.h>
#define LOG_TAG "DINOv3_SAM_Native"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
constexpr int PATCH_SIZE = 16;
constexpr int SAM_INPUT_SIZE = 1024;
constexpr int SAM_MASK_SIZE = 256;

constexpr float MEAN_R = 0.485f * 255.0f;
constexpr float MEAN_G = 0.456f * 255.0f;
constexpr float MEAN_B = 0.406f * 255.0f;
constexpr float STD_R = 0.229f;
constexpr float STD_G = 0.224f;
constexpr float STD_B = 0.225f;

Ort::Env* g_env = nullptr;
Ort::Session* g_dinov3_session = nullptr;
Ort::Session* g_sam_encoder_session = nullptr;
Ort::Session* g_sam_decoder_session = nullptr;

// Helper to preprocess RGB cv::Mat into normalized NCHW ONNX blob for DINOv3
cv::Mat preprocess_image(const cv::Mat& rgb, int input_size, int& out_w_patches, int& out_h_patches) {
    int h_patches = input_size / PATCH_SIZE;
    int w_patches = (rgb.cols * input_size) / (rgb.rows * PATCH_SIZE);
    if (w_patches % 2 != 0) w_patches -= 1;

    out_w_patches = w_patches;
    out_h_patches = h_patches;

    int new_w = w_patches * PATCH_SIZE;
    int new_h = h_patches * PATCH_SIZE;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    cv::Mat blob = cv::dnn::blobFromImage(
        resized,
        1.0 / 255.0,
        cv::Size(new_w, new_h),
        cv::Scalar(MEAN_R, MEAN_G, MEAN_B),
        false,
        false
    );

    cv::Mat ch0(new_h, new_w, CV_32F, blob.ptr<float>(0, 0));
    ch0 /= STD_R;
    cv::Mat ch1(new_h, new_w, CV_32F, blob.ptr<float>(0, 1));
    ch1 /= STD_G;
    cv::Mat ch2(new_h, new_w, CV_32F, blob.ptr<float>(0, 2));
    ch2 /= STD_B;

    return blob;
}

// Helper to preprocess RGB cv::Mat into 1024x1024 NCHW blob for EdgeTAM Encoder
cv::Mat preprocess_sam_image(const cv::Mat& rgb) {
    cv::Mat blob = cv::dnn::blobFromImage(
        rgb,
        1.0 / 255.0,
        cv::Size(SAM_INPUT_SIZE, SAM_INPUT_SIZE),
        cv::Scalar(MEAN_R, MEAN_G, MEAN_B),
        false,
        false
    );

    cv::Mat ch0(SAM_INPUT_SIZE, SAM_INPUT_SIZE, CV_32F, blob.ptr<float>(0, 0));
    ch0 /= STD_R;
    cv::Mat ch1(SAM_INPUT_SIZE, SAM_INPUT_SIZE, CV_32F, blob.ptr<float>(0, 1));
    ch1 /= STD_G;
    cv::Mat ch2(SAM_INPUT_SIZE, SAM_INPUT_SIZE, CV_32F, blob.ptr<float>(0, 2));
    ch2 /= STD_B;

    return blob;
}

// Runs DINOv3 inference on an NCHW blob
std::vector<Ort::Value> run_dinov3_inference(const cv::Mat& blob, int new_w, int new_h) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = g_dinov3_session->GetInputNameAllocated(0, allocator);
    auto output_name_alloc = g_dinov3_session->GetOutputNameAllocated(0, allocator);

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

    return g_dinov3_session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
}

// Runs EdgeTAM Encoder + Decoder using Gemma 4 normalized bounding box [ymin, xmin, ymax, xmax]
// Returns a 256x256 binary mask (CV_8UC1, 255 = foreground, 0 = background)
cv::Mat run_edgetam_box_segmentation(
    const cv::Mat& rgb,
    float ymin,
    float xmin,
    float ymax,
    float xmax
) {
    if (!g_sam_encoder_session || !g_sam_decoder_session) {
        return cv::Mat();
    }

    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // 1. Run EdgeTAM Encoder
        cv::Mat sam_blob = preprocess_sam_image(rgb);
        std::vector<int64_t> enc_input_shape = {1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE};
        Ort::Value enc_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            reinterpret_cast<float*>(sam_blob.data),
            sam_blob.total(),
            enc_input_shape.data(),
            enc_input_shape.size()
        );

        const char* enc_input_names[] = { "image" };
        const char* enc_output_names[] = { "image_embed", "high_res_feats_0", "high_res_feats_1" };

        auto enc_outputs = g_sam_encoder_session->Run(
            Ort::RunOptions{nullptr},
            enc_input_names,
            &enc_input_tensor,
            1,
            enc_output_names,
            3
        );

        // 2. Prepare EdgeTAM Decoder Box Prompt Inputs (labels 2.0 = top-left, 3.0 = bottom-right)
        std::vector<float> point_coords = {
            xmin * SAM_INPUT_SIZE, ymin * SAM_INPUT_SIZE,
            xmax * SAM_INPUT_SIZE, ymax * SAM_INPUT_SIZE
        };
        std::vector<int64_t> coords_shape = {1, 2, 2};

        std::vector<float> point_labels = { 2.0f, 3.0f };
        std::vector<int64_t> labels_shape = {1, 2};

        std::vector<float> empty_mask(1 * 1 * SAM_MASK_SIZE * SAM_MASK_SIZE, 0.0f);
        std::vector<int64_t> mask_input_shape = {1, 1, SAM_MASK_SIZE, SAM_MASK_SIZE};

        std::vector<float> no_mask_flag = { 0.0f };
        std::vector<int64_t> flag_shape = {1};

        std::vector<Ort::Value> dec_inputs;
        dec_inputs.push_back(std::move(enc_outputs[0])); // image_embed
        dec_inputs.push_back(std::move(enc_outputs[1])); // high_res_feats_0
        dec_inputs.push_back(std::move(enc_outputs[2])); // high_res_feats_1
        dec_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, point_coords.data(), point_coords.size(),
            coords_shape.data(), coords_shape.size()
        ));
        dec_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, point_labels.data(), point_labels.size(),
            labels_shape.data(), labels_shape.size()
        ));
        dec_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, empty_mask.data(), empty_mask.size(),
            mask_input_shape.data(), mask_input_shape.size()
        ));
        dec_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, no_mask_flag.data(), no_mask_flag.size(),
            flag_shape.data(), flag_shape.size()
        ));

        const char* dec_input_names[] = {
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input"
        };
        const char* dec_output_names[] = { "low_res_masks", "iou_predictions" };

        auto dec_outputs = g_sam_decoder_session->Run(
            Ort::RunOptions{nullptr},
            dec_input_names,
            dec_inputs.data(),
            dec_inputs.size(),
            dec_output_names,
            2
        );

        // 3. Select highest-IoU mask candidate
        const float* masks_ptr = dec_outputs[0].GetTensorData<float>();
        const float* iou_ptr = dec_outputs[1].GetTensorData<float>();
        size_t num_masks = dec_outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();

        int best_mask_idx = 0;
        for (size_t i = 1; i < num_masks; ++i) {
            if (iou_ptr[i] > iou_ptr[best_mask_idx]) {
                best_mask_idx = static_cast<int>(i);
            }
        }

        // 4. Threshold logits > 0.0f into 256x256 binary mask
        const int mask_area = SAM_MASK_SIZE * SAM_MASK_SIZE;
        const float* best_mask_slice = masks_ptr + (best_mask_idx * mask_area);

        cv::Mat binary_mask(SAM_MASK_SIZE, SAM_MASK_SIZE, CV_8UC1);
        for (int i = 0; i < mask_area; ++i) {
            binary_mask.data[i] = (best_mask_slice[i] > 0.0f) ? 255 : 0;
        }

        // 5. Morphological cleanup (fill holes & remove isolated islands)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel);

        return binary_mask;
    } catch (const Ort::Exception& e) {
        LOGE("EdgeTAM ONNX Exception: %s", e.what());
        return cv::Mat();
    }
}
} // namespace

extern "C" {

__attribute__((visibility("default"))) __attribute__((used))
void init_session(
    const char* dinov3_model_path,
    const char* sam_encoder_path,
    const char* sam_decoder_path
) {
    try {
        delete g_dinov3_session; g_dinov3_session = nullptr;
        delete g_sam_encoder_session; g_sam_encoder_session = nullptr;
        delete g_sam_decoder_session; g_sam_decoder_session = nullptr;
        delete g_env; g_env = nullptr;

        g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DINOv3_SAM_Native");

        // DINOv3 Session (with NNAPI acceleration for real-time frame loop)
        Ort::SessionOptions dino_options;
        dino_options.SetIntraOpNumThreads(4);
        #if defined(__ANDROID__)
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*)dino_options, 0);
        if (status != nullptr) {
            Ort::GetApi().ReleaseStatus(status);
        }
        #endif
        g_dinov3_session = new Ort::Session(*g_env, dinov3_model_path, dino_options);

        // EdgeTAM Encoder & Decoder Sessions (CPU 4-thread to avoid GPU contention with Gemma 4)
        Ort::SessionOptions sam_options;
        sam_options.SetIntraOpNumThreads(4);
        sam_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        g_sam_encoder_session = new Ort::Session(*g_env, sam_encoder_path, sam_options);
        g_sam_decoder_session = new Ort::Session(*g_env, sam_decoder_path, sam_options);

        LOGE("Initialized DINOv3 + EdgeTAM Encoder & Decoder sessions.");
    } catch (const std::exception& e) {
        LOGE("Exception in init_session: %s", e.what());
    }
}

__attribute__((visibility("default"))) __attribute__((used))
float* create_prototype(uint8_t* rgba_data, int length, int input_size, int* out_feature_dim) {
    if (!g_dinov3_session || !rgba_data || length <= 0) {
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
    auto output_tensors = run_dinov3_inference(blob, new_w, new_h);
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
    if (!g_dinov3_session || !yuv_data || !prototype) {
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
    auto output_tensors = run_dinov3_inference(blob, w_patches * PATCH_SIZE, h_patches * PATCH_SIZE);
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
    if (!g_dinov3_session || !yuv_data) {
        *out_feature_dim = 0;
        return nullptr;
    }

    // 1. Convert YUV420 to RGB and rotate 90 degrees clockwise to match portrait camera
    cv::Mat yuv(height * 3 / 2, width, CV_8UC1, yuv_data);
    cv::Mat rgb, rotated;
    cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
    cv::rotate(rgb, rotated, cv::ROTATE_90_CLOCKWISE);

    // 2. Run DINOv3 feature extraction on the rotated frame
    int w_patches = 0, h_patches = 0;
    cv::Mat blob = preprocess_image(rotated, input_size, w_patches, h_patches);
    int new_w = w_patches * PATCH_SIZE;
    int new_h = h_patches * PATCH_SIZE;

    auto output_tensors = run_dinov3_inference(blob, new_w, new_h);
    const float* all_features = output_tensors.front().GetTensorData<float>();
    size_t total_elements = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    int num_patches = w_patches * h_patches;
    int feature_dim = static_cast<int>(total_elements / num_patches);

    // 3. Run EdgeTAM box-prompted segmentation using Gemma 4's bounding box
    cv::Mat sam_mask_256 = run_edgetam_box_segmentation(rotated, ymin, xmin, ymax, xmax);

    std::vector<float> final_proto(feature_dim, 0.0f);
    int fg_count = 0;

    if (!sam_mask_256.empty()) {
        // Downscale 256x256 EdgeTAM mask to DINOv3 patch grid (w_patches x h_patches)
        cv::Mat patch_mask;
        cv::resize(sam_mask_256, patch_mask, cv::Size(w_patches, h_patches), 0, 0, cv::INTER_NEAREST);

        for (int i = 0; i < num_patches; ++i) {
            if (patch_mask.data[i] > 127) {
                const float* patch_feat = all_features + (i * feature_dim);
                for (int d = 0; d < feature_dim; ++d) {
                    final_proto[d] += patch_feat[d];
                }
                fg_count++;
            }
        }
    }

    // 4. Fallback: If EdgeTAM produced an empty mask after downscaling, pool inner box patches
    if (fg_count == 0) {
        int x0 = std::clamp(static_cast<int>(std::floor(xmin * w_patches)), 0, w_patches - 1);
        int x1 = std::clamp(static_cast<int>(std::ceil(xmax * w_patches)) - 1, x0, w_patches - 1);
        int y0 = std::clamp(static_cast<int>(std::floor(ymin * h_patches)), 0, h_patches - 1);
        int y1 = std::clamp(static_cast<int>(std::ceil(ymax * h_patches)) - 1, y0, h_patches - 1);

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float* patch_feat = all_features + ((y * w_patches + x) * feature_dim);
                for (int d = 0; d < feature_dim; ++d) {
                    final_proto[d] += patch_feat[d];
                }
                fg_count++;
            }
        }
    }

    if (fg_count == 0) {
        *out_feature_dim = 0;
        return nullptr;
    }

    for (int d = 0; d < feature_dim; ++d) {
        final_proto[d] /= static_cast<float>(fg_count);
    }

    float* result = new float[feature_dim];
    std::memcpy(result, final_proto.data(), feature_dim * sizeof(float));
    *out_feature_dim = feature_dim;

    return result;
}

} // extern "C"
