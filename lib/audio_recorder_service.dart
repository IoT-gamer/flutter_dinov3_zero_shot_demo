import 'dart:async';

import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';

class AudioRecorderService {
  final AudioRecorder _audioRecorder = AudioRecorder();
  Timer? _recordTimer;

  // Starts recording and automatically stops after the duration limit
  Future<void> startRecordingWithLimit({
    int timeLimitSeconds = 10,
    required Function(String?) onRecordingComplete,
  }) async {
    if (!await _audioRecorder.hasPermission()) {
      onRecordingComplete(null);
      return;
    }

    final directory = await getTemporaryDirectory();

    final audioPath = '${directory.path}/inference_audio.wav';

    // Use WAV, Mono, 16kHz
    await _audioRecorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: audioPath,
    );

    _recordTimer = Timer(Duration(seconds: timeLimitSeconds), () async {
      await stopRecording(onRecordingComplete);
    });
  }

  // Stops recording manually or via the Timer callback
  Future<void> stopRecording(Function(String?) onRecordingComplete) async {
    _recordTimer?.cancel();
    final path = await _audioRecorder.stop();
    onRecordingComplete(path);
  }

  void dispose() {
    _recordTimer?.cancel();
    _audioRecorder.dispose();
  }
}
