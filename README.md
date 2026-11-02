# Gemma 4 + DINOv3 Zero-Shot Segmentation & Tracking (Flutter Edge AI)

> ⚠️ **Platform Notice:** Currently, only **Android** is supported. This application uses a hybrid native architecture combining Android JNI (`LiteRT-LM`) and C++ FFI (`ONNX Runtime` + `OpenCV`).

A Flutter Edge AI application demonstrating **on-device, zero-shot multimodal object segmentation and real-time tracking**. By combining **Gemma 4 (`gemma-4-E2B-it.litertlm`)** via Google's LiteRT-LM with **DINOv3** via ONNX Runtime and OpenCV in C++, users can simply **speak** or **type** the name of any object in the camera view to automatically ground, refine, and segment it in real time—with zero cloud dependencies.

---

## 📷 Screenshots
<img src="docs/images/1.jpg" alt="drawing" height="200"/> <img src="docs/images/2.jpg" alt="drawing" height="200"/>

## 🧠 System 1 / System 2 Edge AI Architecture

This repository implements a **Slow-Fast Visual Pipeline** that partitions workloads across your Android device's hardware accelerators:

| Component | Model | Runtime & Bridge | Compute Target | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **System 2 (Slow / On-Demand)** | **Gemma 4 E2B** (`model.litertlm`) | LiteRT-LM via **`jnigen` JNI** (`LitertBridge.kt`) | **GPU** (Vision & Language w/ Speculative Decoding) + **CPU** (Audio) | **Zero-Shot Multimodal Grounding:** Processes a single camera frame + voice WAV (16kHz mono) or text prompt to locate the target `[ymin, xmin, ymax, xmax]`. |
| **Refinement Bridge** | **DINOv3 Center-Seed Filter** | C++ (`native_segmentation.cpp`) via **Dart FFI** | **CPU SIMD** | **Zero-Overhead Mask Refinement:** Extracts a semantic seed vector from the inner 36% core of Gemma 4's box and filters out background corner patches via cosine similarity ($\ge 0.60$). |
| **System 1 (Fast Loop)** | **DINOv3** (`dinov3_feature_extractor.onnx`) | ONNX Runtime `1.30.0` + OpenCV `5.0.0` via **Dart FFI** (`StatefulIsolate`) | **NNAPI (NPU)** / Multi-thread CPU | **Real-Time Tracking (15–30 FPS):** Computes dense patch features, cosine similarity masks, and largest connected-component filtering on live `YUV420` camera frames. |

---

## ✨ Key Features

* **Zero-Shot Voice & Text Enrollment:** Point the camera at an object and say *"coffee mug"* (via a 5-second 16kHz mono WAV recording) or type a description. No pre-cut images required.
* **DINOv3 Center-Seeded Feature Refinement:** Eliminates the need for a third segmentation model (like SAM) in RAM by using DINOv3's own patch embeddings inside the core of Gemma 4's bounding box to reject background noise before building the object prototype.
* **Fallback 1-Shot Gallery Enrollment:** Support for uploading a pre-masked RGBA PNG with transparency from the device gallery.
* **Zero-Copy Native Camera Pipeline:** Raw `YUV420` camera planes are passed directly to C++ via FFI for OpenCV SIMD rotation, normalization, and ONNX inference inside a background `StatefulIsolate`.
* **Multi-Token Prediction (MTP):** Enables experimental speculative decoding on the LiteRT-LM GPU backend for fast Gemma 4 coordinate generation.
* **Interactive Controls:**
  * Live **Cosine Similarity Threshold Slider** (`0.50` – `0.90`).
  * Dynamic **DINOv3 Input Resolution Selector** (`320`, `400`, `512`, `768`).
  * **Largest Area Filtering** via OpenCV `cv::connectedComponentsWithStats` to suppress background noise.
  * **Visual Grounding Overlay** displaying Gemma 4's detected bounding box (cyan) and the inner 36% DINOv3 seed core (amber).

---

## 📁 Repository Structure

```text
android/
  └── app/src/main/
      ├── cpp/
      │   ├── CMakeLists.txt           # Fetches ONNX Runtime 1.30.0 & OpenCV 5.0.0 SDK
      │   └── native_segmentation.cpp  # DINOv3 inference, Center-Seed refinement & YUV-to-JPEG
      └── kotlin/.../
          ├── LitertBridge.kt          # Native LiteRT-LM wrapper (Vision, Audio & Grounding)
          └── MainActivity.kt          # Flutter entry point
lib/
  ├── cubit/
  │   ├── segmentation_cubit.dart      # State management orchestrating Gemma 4 & DINOv3 isolate
  │   └── segmentation_state.dart      # Immutable state (prototype, overlay mask, bounding box)
  ├── src/generated/
  │   └── litertlm_bindings.dart       # Auto-generated JNI bindings (via jnigen)
  ├── audio_recorder_service.dart      # 16kHz mono PCM WAV recorder for LiteRT-LM audio
  ├── camera_screen.dart               # Live camera preview, voice/text bar & custom painters
  ├── constants.dart                   # Model asset constants
  ├── inference_service.dart           # Memory-safe Dart wrapper around JNI LitertBridge
  ├── main.dart                        # App entry, camera setup & LiteRT-LM initialization
  └── segmentation_isolate.dart        # Dart FFI bindings & background isolate workers
tool/
  └── jnigen.dart                      # JNI binding generator script
```

---

## ⚙️ Prerequisites

1. **Flutter SDK:** `^3.10.0` (or `^3.11.4` recommended)
2. **Java 21 JDK:** Required by `litertlm-android`. Configure Flutter to use JDK 21:
   ```bash
   flutter config --jdk-dir="/usr/lib/jvm/java-21-openjdk-amd64"
   ```
3. **CMake & Android NDK:** Installed via Android Studio SDK Manager (CMake automatically downloads OpenCV 5.0.0 and ONNX Runtime 1.30.0 during the Gradle build).
4. **Physical Android Device:** Modern SoC with GPU/NPU support and at least 8 GB RAM recommended.
5. **Models Required:**
   * **DINOv3 ONNX Model (`dinov3_feature_extractor.onnx`):** Export using the [DINOv3 ONNX Export Notebook](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/blob/main/notebooks/dinov3_onnx_export.ipynb) and place it in `assets/dinov3_feature_extractor.onnx`.
   * **Gemma 4 LiteRT-LM Model (`gemma-4-E2B-it.litertlm`):** Download from [Hugging Face (`litert-community/gemma-4-E2B-it-litert-lm`)](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main).

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd flutter_dinov3_gemma4_zero_shot
flutter pub get
```

### 2. Place the DINOv3 ONNX Asset
Ensure your exported DINOv3 model is located at:
```text
assets/dinov3_feature_extractor.onnx
```

### 3. Build & Regenerate JNI Bindings (If Modifying `LitertBridge.kt`)
Pre-generated bindings are located in `lib/src/generated/litertlm_bindings.dart`. If you change the package name or methods in `LitertBridge.kt`, compile the debug APK first so `jnigen` can inspect the compiled Kotlin bytecode:
```bash
flutter build apk --debug
dart run tool/jnigen.dart
```
*If the `LitertBridge` class cannot be found, you may need to modify `classPath` in `tool/jnigen.dart`. To find the correct paths, you can check your `build` directory after a successful build. In macOS/Linux run: `find build/app -name "LitertBridge.class"`*

### 4. Run the App & Push the Gemma 4 Model
First, launch the app once on your connected Android device so Android creates the app's external storage directory:
```bash
flutter run
```
*(On the first launch, the app will display a startup message instructing you to push `model.litertlm` via ADB.)*

Push the downloaded `gemma-4-E2B-it.litertlm` file to the device:
```bash
adb push gemma-4-E2B-it.litertlm /storage/emulated/0/Android/data/com.example.flutter_dinov3_zero_shot_demo/files/model.litertlm
```
Restart the app—Gemma 4 will load into GPU memory and DINOv3 will initialize in the background isolate.

---

## 📱 How to Use the App

### Method A: Zero-Shot Voice or Text Enrollment (Gemma 4 -> DINOv3)
1. Point your camera at the target object you want to segment and track.
2. **By Text:** Type the object's description in the top bar (e.g., *"red coffee mug"* or *"soldering iron"*) and tap the **Send** icon.
3. **By Voice:** Tap the **Microphone** icon in the top bar and speak the name of the object. Tap the stop icon (or wait 5 seconds) to submit.
4. **Automatic Hand-off:**
   * Live segmentation pauses briefly to dedicate GPU bandwidth to Gemma 4.
   * OpenCV saves a 700px snapshot (**280 Token Budget**) of the current camera frame.
   * Gemma 4 returns the bounding box `[ymin, xmin, ymax, xmax]`.
   * C++ runs **DINOv3 Center-Seeded Refinement** inside the box and automatically starts real-time tracking!

### Method B: One-Shot RGBA PNG Enrollment (Fallback)
1. Tap the **Upload File** floating action button.
2. Select a 4-channel RGBA PNG with a transparent background from your gallery. (can use [flutter_segment_anything_app](https://github.com/IoT-gamer/flutter_segment_anything_app) to generate mask)
3. Tap the **Play** button to start real-time segmentation.

### Fine-Tuning Tracking Performance
* **Adjust Sensitivity (`Icons.tune`):** Tap the Tune icon in the top-right app bar to reveal the cosine similarity slider (`0.50` to `0.90`).
* **Filter Largest Object (`Icons.filter_center_focus`):** Tap the bottom floating action button to keep only the largest contiguous segmented region via OpenCV connected components.
* **Change DINOv3 Resolution (`Icons.aspect_ratio`):** Choose between `320` (fastest FPS), `400`, `512`, or `768` (finest patch grid). *Note: Changing resolution clears the active prototype.*

---

## 🙏 Acknowledgements

* **DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
* **Google LiteRT-LM:** [Google AI Edge LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview)
* **Gemma 4 E2B LiteRT-LM Weights:** [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
* **Dart JNI Generator:** [jnigen](https://pub.dev/packages/jnigen)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.