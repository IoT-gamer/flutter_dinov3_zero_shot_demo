import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:integral_isolates/integral_isolates.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../constants.dart';
import '../inference_service.dart';
import '../segmentation_isolate.dart';

part 'segmentation_state.dart';

class SegmentationCubit extends Cubit<SegmentationState> {
  SegmentationCubit() : super(const SegmentationState());

  late final StatefulIsolate _isolate;
  bool _isModelReady = false;
  bool _isProcessing = false;

  // Cached latest camera frame for zero-shot Gemma 4 grounding
  List<Uint8List>? _latestPlanes;
  int _latestWidth = 0;
  int _latestHeight = 0;

  Future<void> initialize() async {
    emit(state.copyWith(status: SegmentationStatus.loadingModel));
    _isolate = StatefulIsolate();
    try {
      final byteData = await rootBundle.load(
        'assets/${AppConstants.modelFilename}',
      );
      final directory = await getApplicationDocumentsDirectory();
      final modelPath = p.join(directory.path, AppConstants.modelFilename);
      await File(modelPath).writeAsBytes(byteData.buffer.asUint8List());
      final token = RootIsolateToken.instance!;
      _isModelReady = await _isolate.compute(initializeSession, {
        'path': modelPath,
        'token': token,
      });
      emit(state.copyWith(status: SegmentationStatus.modelReady));
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'Failed to initialize model: $e',
        ),
      );
    }
  }

  /// Fallback 1-shot enrollment: Pick a pre-masked RGBA PNG from the gallery
  Future<void> setReferencePrototype() async {
    if (!_isModelReady ||
        state.status == SegmentationStatus.creatingPrototype) {
      return;
    }
    emit(
      state.copyWith(
        status: SegmentationStatus.creatingPrototype,
        isSegmenting: false,
        clearOverlay: true,
        clearBox: true,
      ),
    );

    try {
      final XFile? pickedFile = await ImagePicker().pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile == null) {
        emit(state.copyWith(status: SegmentationStatus.modelReady));
        return;
      }

      final fileBytes = await pickedFile.readAsBytes();
      final image = img.decodePng(fileBytes);
      if (image == null || image.numChannels != 4) {
        emit(
          state.copyWith(
            status: SegmentationStatus.error,
            errorMessage:
                'Invalid file. Please select a PNG with transparency (RGBA).',
          ),
        );
        emit(
          state.copyWith(
            status: SegmentationStatus.modelReady,
            errorMessage: null,
          ),
        );
        return;
      }

      final prototype = await _isolate.compute(createPrototype, {
        'bytes': fileBytes,
        'inputSize': state.selectedInputSize,
      });

      if (prototype.isNotEmpty) {
        emit(
          state.copyWith(
            status: SegmentationStatus.prototypeReady,
            objectPrototype: prototype,
            clearBox: true,
          ),
        );
      } else {
        emit(
          state.copyWith(
            status: SegmentationStatus.error,
            errorMessage: 'Could not create prototype from the selected image.',
          ),
        );
        emit(
          state.copyWith(
            status: SegmentationStatus.modelReady,
            errorMessage: null,
          ),
        );
      }
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'An error occurred: $e',
        ),
      );
      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          errorMessage: null,
        ),
      );
    }
  }

  /// Zero-shot enrollment via Gemma 4 (Voice WAV or Text Prompt) + DINOv3 Center-Seed Refinement
  Future<void> setPrototypeFromGemma({
    required InferenceService gemmaService,
    String? textTarget,
    String? audioPath,
  }) async {
    if (!_isModelReady ||
        state.status == SegmentationStatus.creatingPrototype) {
      return;
    }

    if (_latestPlanes == null || _latestWidth == 0 || _latestHeight == 0) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'Camera frame not ready yet. Point camera at target.',
        ),
      );
      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          errorMessage: null,
        ),
      );
      return;
    }

    // Pause live DINOv3 segmentation so GPU/CPU resources are dedicated to Gemma 4
    emit(
      state.copyWith(
        status: SegmentationStatus.creatingPrototype,
        isSegmenting: false,
        clearOverlay: true,
        clearBox: true,
      ),
    );

    // Yield briefly so Flutter renders the progress indicator before JNI call blocks
    await Future.delayed(const Duration(milliseconds: 50));

    try {
      // Snapshot the exact camera frame at the moment of the voice/text request
      final frozenPlanes = _latestPlanes!
          .map((plane) => Uint8List.fromList(plane))
          .toList();
      final frozenW = _latestWidth;
      final frozenH = _latestHeight;

      final tempDir = await getTemporaryDirectory();
      final framePath = p.join(tempDir.path, 'gemma_grounding_frame.jpg');

      // Rotate & resize YUV420 to JPEG in C++ (700px = 280 Gemma Token Budget)
      final saved = await _isolate.compute(saveFrameForGemma, {
        'planes': frozenPlanes,
        'width': frozenW,
        'height': frozenH,
        'maxDim': 700,
        'outPath': framePath,
      });

      if (!saved) {
        throw Exception('Failed to save camera frame for Gemma 4.');
      }

      // Query Gemma 4 for bounding box coordinates [ymin, xmin, ymax, xmax]
      final rawResponse = await gemmaService.groundObject(
        imagePath: framePath,
        audioPath: audioPath,
        textPrompt: textTarget,
      );

      final box = _parseGemmaBox(rawResponse);
      if (box == null) {
        throw Exception('Object not found (Gemma returned: "$rawResponse")');
      }

      // Extract DINOv3 prototype using Center-Seeded Refinement on the same frame
      final prototype = await _isolate.compute(createPrototypeFromBox, {
        'planes': frozenPlanes,
        'width': frozenW,
        'height': frozenH,
        'box': box,
        'inputSize': state.selectedInputSize,
      });

      if (prototype.isNotEmpty) {
        emit(
          state.copyWith(
            status: SegmentationStatus.prototypeReady,
            objectPrototype: prototype,
            detectedBox: box,
            isSegmenting: true, // Automatically begin real-time tracking
          ),
        );
      } else {
        throw Exception('Failed to extract DINOv3 features from bounding box.');
      }
    } catch (e) {
      emit(
        state.copyWith(status: SegmentationStatus.error, errorMessage: '$e'),
      );
      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          errorMessage: null,
        ),
      );
    }
  }

  /// Parses Gemma's [ymin, xmin, ymax, xmax] (0..1000 scale) into normalized [0.0..1.0] floats
  List<double>? _parseGemmaBox(String response) {
    final match = RegExp(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
        .firstMatch(response);
    if (match == null) return null;

    final ymin = (int.parse(match.group(1)!) / 1000.0).clamp(0.0, 1.0);
    final xmin = (int.parse(match.group(2)!) / 1000.0).clamp(0.0, 1.0);
    final ymax = (int.parse(match.group(3)!) / 1000.0).clamp(0.0, 1.0);
    final xmax = (int.parse(match.group(4)!) / 1000.0).clamp(0.0, 1.0);

    if (ymax <= ymin || xmax <= xmin) return null;
    return [ymin, xmin, ymax, xmax];
  }

  void processCameraImage(CameraImage cameraImage) {
    // Always keep a lightweight reference to the latest YUV420 planes for enrollment
    if (cameraImage.format.group == ImageFormatGroup.yuv420 &&
        state.status != SegmentationStatus.creatingPrototype) {
      _latestPlanes = cameraImage.planes.map((p) => p.bytes).toList();
      _latestWidth = cameraImage.width;
      _latestHeight = cameraImage.height;
    }

    if (!state.isSegmenting ||
        _isProcessing ||
        state.objectPrototype == null ||
        !_isModelReady) {
      return;
    }
    _isProcessing = true;

    _isolate
        .compute(runSegmentation, {
          'prototype': state.objectPrototype!,
          'planes': cameraImage.planes.map((p) => p.bytes).toList(),
          'width': cameraImage.width,
          'height': cameraImage.height,
          'format': cameraImage.format.group,
          'threshold': state.similarityThreshold,
          'inputSize': state.selectedInputSize,
          'showLargestOnly': state.showLargestAreaOnly,
        })
        .then((result) {
          if (state.isSegmenting && result.isNotEmpty) {
            _updateOverlay(
              result['scores'] as List<double>,
              result['width'] as int,
              result['height'] as int,
            );
          }
          _isProcessing = false;
        });
  }

  void _updateOverlay(List<double> scores, int width, int height) {
    final pixels = Uint8List(width * height * 4);
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > state.similarityThreshold) {
        pixels[i * 4 + 0] = 30; // R
        pixels[i * 4 + 1] = 255; // G
        pixels[i * 4 + 2] = 150; // B
        pixels[i * 4 + 3] = 170; // A
      }
    }

    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (
      img,
    ) {
      emit(state.copyWith(overlayImage: img));
    });
  }

  void toggleSegmentation() {
    if (state.status != SegmentationStatus.prototypeReady) return;
    final newIsSegmenting = !state.isSegmenting;
    emit(
      state.copyWith(
        isSegmenting: newIsSegmenting,
        clearOverlay: !newIsSegmenting,
      ),
    );
  }

  void updateInputSize(int newSize) {
    if (newSize != state.selectedInputSize) {
      emit(
        state.copyWith(
          selectedInputSize: newSize,
          isSegmenting: false,
          status: SegmentationStatus.modelReady,
          clearPrototype: true,
          clearOverlay: true,
          clearBox: true,
        ),
      );
    }
  }

  void updateThreshold(double value) =>
      emit(state.copyWith(similarityThreshold: value));
  void toggleLargestAreaOnly() =>
      emit(state.copyWith(showLargestAreaOnly: !state.showLargestAreaOnly));
  void toggleSliderVisibility() =>
      emit(state.copyWith(showSlider: !state.showSlider));

  @override
  Future<void> close() {
    _isolate.dispose();
    return super.close();
  }
}
