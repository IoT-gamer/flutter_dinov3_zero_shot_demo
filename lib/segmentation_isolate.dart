import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';

// --- FFI Type Definitions ---

// init_session(const char* dinov3_path, const char* sam_encoder_path, const char* sam_decoder_path)
typedef InitSessionC = Void Function(
  Pointer<Utf8> dinov3Path,
  Pointer<Utf8> samEncoderPath,
  Pointer<Utf8> samDecoderPath,
);
typedef InitSessionDart = void Function(
  Pointer<Utf8> dinov3Path,
  Pointer<Utf8> samEncoderPath,
  Pointer<Utf8> samDecoderPath,
);

// create_prototype(uint8_t* rgba_data, int length, int input_size, int* out_feature_dim)
typedef CreatePrototypeC = Pointer<Float> Function(
  Pointer<Uint8> rgbaData,
  Int32 length,
  Int32 inputSize,
  Pointer<Int32> outFeatureDim,
);
typedef CreatePrototypeDart = Pointer<Float> Function(
  Pointer<Uint8> rgbaData,
  int length,
  int inputSize,
  Pointer<Int32> outFeatureDim,
);

// run_segmentation(uint8_t* yuv_data, int width, int height, float* prototype, int feature_dim, double threshold, int input_size, bool largest_only, int* out_w, int* out_h)
typedef RunSegmentationC = Pointer<Float> Function(
  Pointer<Uint8> yuvData,
  Int32 width,
  Int32 height,
  Pointer<Float> prototype,
  Int32 featureDim,
  Double threshold,
  Int32 inputSize,
  Bool largestOnly,
  Pointer<Int32> outW,
  Pointer<Int32> outH,
);
typedef RunSegmentationDart = Pointer<Float> Function(
  Pointer<Uint8> yuvData,
  int width,
  int height,
  Pointer<Float> prototype,
  int featureDim,
  double threshold,
  int inputSize,
  bool largestOnly,
  Pointer<Int32> outW,
  Pointer<Int32> outH,
);

// free_pointer(void* ptr) - Essential to prevent memory leaks from C++ allocations
typedef FreePointerC = Void Function(Pointer<Void> ptr);
typedef FreePointerDart = void Function(Pointer<Void> ptr);

typedef SaveYuvForGemmaC = Bool Function(
  Pointer<Uint8> yuvData,
  Int32 width,
  Int32 height,
  Int32 maxDim,
  Pointer<Utf8> outJpegPath,
);
typedef SaveYuvForGemmaDart = bool Function(
  Pointer<Uint8> yuvData,
  int width,
  int height,
  int maxDim,
  Pointer<Utf8> outJpegPath,
);

typedef CreatePrototypeFromBoxC = Pointer<Float> Function(
  Pointer<Uint8> yuvData,
  Int32 width,
  Int32 height,
  Float ymin,
  Float xmin,
  Float ymax,
  Float xmax,
  Int32 inputSize,
  Pointer<Utf8> outCutoutPngPath,
  Pointer<Int32> outFeatureDim,
);

typedef CreatePrototypeFromBoxDart = Pointer<Float> Function(
  Pointer<Uint8> yuvData,
  int width,
  int height,
  double ymin,
  double xmin,
  double ymax,
  double xmax,
  int inputSize,
  Pointer<Utf8> outCutoutPngPath,
  Pointer<Int32> outFeatureDim,
);

// --- Native Library Setup ---

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libdinov3_native.so')
    : DynamicLibrary.process();

final initSessionNative = nativeLib
    .lookupFunction<InitSessionC, InitSessionDart>('init_session');
final createPrototypeNative = nativeLib
    .lookupFunction<CreatePrototypeC, CreatePrototypeDart>('create_prototype');
final runSegmentationNative = nativeLib
    .lookupFunction<RunSegmentationC, RunSegmentationDart>('run_segmentation');
final freePointerNative = nativeLib
    .lookupFunction<FreePointerC, FreePointerDart>('free_pointer');

final saveYuvForGemmaNative = nativeLib
    .lookupFunction<SaveYuvForGemmaC, SaveYuvForGemmaDart>(
      'save_yuv_for_gemma',
    );
final createPrototypeFromBoxNative = nativeLib
    .lookupFunction<CreatePrototypeFromBoxC, CreatePrototypeFromBoxDart>(
      'create_prototype_from_box',
    );

// --- Isolate Functions ---

/// Returns a boolean indicating success.
Future<bool> initializeSession(Map<String, dynamic> args) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(
    args['token'] as RootIsolateToken,
  );

  final dinoPtr = (args['dinov3Path'] as String).toNativeUtf8();
  final encPtr = (args['samEncoderPath'] as String).toNativeUtf8();
  final decPtr = (args['samDecoderPath'] as String).toNativeUtf8();

  initSessionNative(dinoPtr, encPtr, decPtr);

  malloc.free(dinoPtr);
  malloc.free(encPtr);
  malloc.free(decPtr);

  return true;
}

// Helper to pack CameraImage YUV420 planes into a contiguous planar I420 buffer
Pointer<Uint8> _packYuvPlanes(
  List<Uint8List> planes,
  int width,
  int height, {
  List<int>? bytesPerRow,
  List<int?>? bytesPerPixel,
}) {
  final int ySize = width * height;
  final int uvWidth = width ~/ 2;
  final int uvHeight = height ~/ 2;
  final int uvSize = uvWidth * uvHeight;
  final int yuvSize = ySize + (uvSize * 2);

  final Pointer<Uint8> yuvPointer = malloc.allocate<Uint8>(yuvSize);
  final Uint8List yuvList = yuvPointer.asTypedList(yuvSize);

  final Uint8List yPlane = planes[0];
  final Uint8List uPlane = planes[1];
  final Uint8List vPlane = planes[2];

  final int yRowStride = (bytesPerRow != null && bytesPerRow.isNotEmpty)
      ? bytesPerRow[0]
      : width;
  final int uvPixelStride = (bytesPerPixel != null && bytesPerPixel.length > 1)
      ? (bytesPerPixel[1] ?? 1)
      : (uPlane.length > uvSize ? 2 : 1);
  final int uvRowStride = (bytesPerRow != null && bytesPerRow.length > 1)
      ? bytesPerRow[1]
      : (uvPixelStride == 2 ? width : uvWidth);

  // Copy Y plane (stripping any row padding)
  if (yRowStride == width && yPlane.length >= ySize) {
    yuvList.setRange(0, ySize, yPlane);
  } else {
    int dstY = 0;
    for (int row = 0; row < height; row++) {
      final int srcY = row * yRowStride;
      yuvList.setRange(dstY, dstY + width, yPlane, srcY);
      dstY += width;
    }
  }

  // Copy & de-interleave U and V planes into planar I420
  int dstU = ySize;
  int dstV = ySize + uvSize;

  if (uvPixelStride == 1 && uvRowStride == uvWidth) {
    yuvList.setRange(dstU, dstU + uvSize, uPlane);
    yuvList.setRange(dstV, dstV + uvSize, vPlane);
  } else {
    for (int row = 0; row < uvHeight; row++) {
      final int rowStart = row * uvRowStride;
      for (int col = 0; col < uvWidth; col++) {
        final int uvIndex = rowStart + (col * uvPixelStride);
        yuvList[dstU++] = uPlane[uvIndex];
        yuvList[dstV++] = vPlane[uvIndex];
      }
    }
  }

  return yuvPointer;
}

// Isolate function 1: Save YUV camera frame to JPEG for Gemma 4
Future<bool> saveFrameForGemma(Map<String, dynamic> args) async {
  final List<Uint8List> planes = args['planes'];
  final int width = args['width'];
  final int height = args['height'];
  final int maxDim = args['maxDim'] ?? 700; // 700px = 280 Token Budget
  final String outPath = args['outPath'];

  final yuvPointer = _packYuvPlanes(
    planes,
    width,
    height,
    bytesPerRow: (args['bytesPerRow'] as List?)?.cast<int>(),
    bytesPerPixel: (args['bytesPerPixel'] as List?)?.cast<int?>(),
  );
  final pathPointer = outPath.toNativeUtf8();

  final success = saveYuvForGemmaNative(
    yuvPointer,
    width,
    height,
    maxDim,
    pathPointer,
  );

  malloc.free(yuvPointer);
  malloc.free(pathPointer);
  return success;
}

// Isolate function 2: Create DINOv3 prototype from Gemma 4 bounding box
Future<List<double>> createPrototypeFromBox(Map<String, dynamic> args) async {
  final List<Uint8List> planes = args['planes'];
  final int width = args['width'];
  final int height = args['height'];
  final List<double> box = args['box'];
  final int inputSize = args['inputSize'];
  final String cutoutPath = args['cutoutPath'] ?? '';

  final yuvPointer = _packYuvPlanes(
    planes,
    width,
    height,
    bytesPerRow: (args['bytesPerRow'] as List?)?.cast<int>(),
    bytesPerPixel: (args['bytesPerPixel'] as List?)?.cast<int?>(),
  );
  final cutoutPathPointer = cutoutPath.toNativeUtf8();
  final outDimPointer = malloc.allocate<Int32>(sizeOf<Int32>());

  final Pointer<Float> prototypePointer = createPrototypeFromBoxNative(
    yuvPointer,
    width,
    height,
    box[0],
    box[1],
    box[2],
    box[3],
    inputSize,
    cutoutPathPointer,
    outDimPointer,
  );

  final int featureDim = outDimPointer.value;
  List<double> objectPrototype = [];

  if (featureDim > 0 && prototypePointer != nullptr) {
    objectPrototype = prototypePointer.asTypedList(featureDim).toList();
    freePointerNative(prototypePointer.cast<Void>());
  }

  malloc.free(yuvPointer);
  malloc.free(cutoutPathPointer);
  malloc.free(outDimPointer);
  return objectPrototype;
}

Future<List<double>> createPrototype(Map<String, dynamic> args) async {
  final Uint8List rgbaBytes = args['bytes'];
  final int inputSize = args['inputSize'];

  // Allocate memory for the RGBA bytes
  final Pointer<Uint8> rgbaPointer = malloc.allocate<Uint8>(rgbaBytes.length);
  rgbaPointer.asTypedList(rgbaBytes.length).setAll(0, rgbaBytes);

  // Pointer to receive the output dimension size from C++
  final outDimPointer = malloc.allocate<Int32>(sizeOf<Int32>());

  // Call native C++ function
  final Pointer<Float> prototypePointer = createPrototypeNative(
    rgbaPointer,
    rgbaBytes.length,
    inputSize,
    outDimPointer,
  );

  final int featureDim = outDimPointer.value;
  List<double> objectPrototype = [];

  if (featureDim > 0 && prototypePointer != nullptr) {
    // Copy native memory to Dart list
    objectPrototype = prototypePointer.asTypedList(featureDim).toList();
    // Free the C++ allocated float array
    freePointerNative(prototypePointer.cast<Void>());
  }

  malloc.free(rgbaPointer);
  malloc.free(outDimPointer);

  print('✅ Native Prototype created in Isolate.');
  return objectPrototype;
}

Future<Map<String, dynamic>> runSegmentation(Map<String, dynamic> args) async {
  final List<double> objectPrototype = args['prototype'];
  final List<Uint8List> planes = args['planes'];
  final ImageFormatGroup format = args['format'] ?? ImageFormatGroup.yuv420;
  final int width = args['width'];
  final int height = args['height'];
  final double similarityThreshold = args['threshold'] ?? 0.7;
  final int inputSize = args['inputSize'];
  final bool showLargestOnly = args['showLargestOnly'] ?? false;

  if (format != ImageFormatGroup.yuv420) {
    print('Error: Native pipeline currently expects YUV420 format.');
    return {};
  }

  // Correctly assemble YUV planes to match exact height * 1.5 * width I420 size
  final int yuvSize = width * height * 3 ~/ 2;
  final yuvPointer = _packYuvPlanes(
    planes,
    width,
    height,
    bytesPerRow: (args['bytesPerRow'] as List?)?.cast<int>(),
    bytesPerPixel: (args['bytesPerPixel'] as List?)?.cast<int?>(),
  );
  final yuvList = yuvPointer.asTypedList(yuvSize);

  final int ySize = width * height;
  final int uvSize = ySize ~/ 4;

  // Y plane
  yuvList.setRange(0, ySize, planes[0]);

  // U plane (handle plane length variations safely)
  final uPlane = planes[1];
  yuvList.setRange(
    ySize,
    ySize + uvSize,
    uPlane.length >= uvSize ? uPlane.sublist(0, uvSize) : uPlane,
  );

  // V plane (handle plane length variations safely)
  final vPlane = planes[2];
  yuvList.setRange(
    ySize + uvSize,
    yuvSize,
    vPlane.length >= uvSize ? vPlane.sublist(0, uvSize) : vPlane,
  );

  // Allocate and pack the prototype feature vector
  final Pointer<Float> prototypePointer = malloc.allocate<Float>(
    objectPrototype.length * sizeOf<Float>(),
  );
  prototypePointer
      .asTypedList(objectPrototype.length)
      .setAll(0, objectPrototype);

  // Pointers to receive output width and height from C++
  final outW = malloc.allocate<Int32>(sizeOf<Int32>());
  final outH = malloc.allocate<Int32>(sizeOf<Int32>());

  // Run native segmentation (Preprocessing, ONNX inference, Cosine Similarity, and Connected Components all happen in C++)
  final Pointer<Float> scoresPointer = runSegmentationNative(
    yuvPointer,
    width,
    height,
    prototypePointer,
    objectPrototype.length,
    similarityThreshold,
    inputSize,
    showLargestOnly,
    outW,
    outH,
  );

  final int wPatches = outW.value;
  final int hPatches = outH.value;
  final int numPatches = wPatches * hPatches;

  List<double> finalScores = [];
  if (numPatches > 0 && scoresPointer != nullptr) {
    finalScores = scoresPointer.asTypedList(numPatches).toList();
    freePointerNative(scoresPointer.cast<Void>());
  }

  // Cleanup memory
  malloc.free(yuvPointer);
  malloc.free(prototypePointer);
  malloc.free(outW);
  malloc.free(outH);

  return {'scores': finalScores, 'width': wPatches, 'height': hPatches};
}
