import 'package:jni/jni.dart';
import 'package:jni_flutter/jni_flutter.dart';

import 'src/generated/litertlm_bindings.dart';

class InferenceService {
  late LitertBridge _bridge;

  void initializeEngine(String modelPath) {
    final androidContext = androidApplicationContext;
    final jModelPath = modelPath.toJString();

    try {
      _bridge = LitertBridge(androidContext, jModelPath);
    } finally {
      jModelPath.release();
    }
  }

  /// Multimodal zero-shot object grounding (Camera Frame + Voice WAV or Text Prompt).
  /// Returns raw model output containing [ymin, xmin, ymax, xmax] scaled 0..1000.
  Future<String> groundObject({
    required String imagePath,
    String? audioPath,
    String? textPrompt,
  }) async {
    final jImagePath = imagePath.toJString();
    final jAudioPath = (audioPath ?? '').toJString();
    final jTextPrompt = (textPrompt ?? '').toJString();

    try {
      final result = _bridge.groundObject(jImagePath, jAudioPath, jTextPrompt);
      final dartResult = result.toDartString();
      result.release();
      return dartResult;
    } finally {
      jImagePath.release();
      jAudioPath.release();
      jTextPrompt.release();
    }
  }

  Future<String> analyzeImage(String prompt, String imagePath) async {
    final jPrompt = prompt.toJString();
    final jPath = imagePath.toJString();

    try {
      final result = _bridge.runVisionInference(jPrompt, jPath);
      final dartResult = result.toDartString();
      result.release();
      return dartResult;
    } finally {
      jPrompt.release();
      jPath.release();
    }
  }

  Future<String> analyzeAudio(String prompt, String audioPath) async {
    final jPrompt = prompt.toJString();
    final jPath = audioPath.toJString();

    try {
      final result = _bridge.runAudioInference(jPrompt, jPath);
      final dartResult = result.toDartString();
      result.release();
      return dartResult;
    } finally {
      jPrompt.release();
      jPath.release();
    }
  }

  void dispose() {
    _bridge.close();
  }
}
