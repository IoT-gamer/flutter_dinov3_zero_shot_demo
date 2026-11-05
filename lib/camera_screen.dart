import 'dart:async';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'audio_recorder_service.dart';
import 'cubit/segmentation_cubit.dart';
import 'inference_service.dart';

class CameraScreen extends StatelessWidget {
  const CameraScreen({
    super.key,
    required this.camera,
    required this.inferenceService,
  });

  final CameraDescription camera;
  final InferenceService inferenceService;

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => SegmentationCubit()..initialize(),
      child: CameraView(camera: camera, inferenceService: inferenceService),
    );
  }
}

class CameraView extends StatefulWidget {
  const CameraView({
    super.key,
    required this.camera,
    required this.inferenceService,
  });

  final CameraDescription camera;
  final InferenceService inferenceService;

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  final TextEditingController _promptController = TextEditingController();
  final AudioRecorderService _audioService = AudioRecorderService();

  bool _isRecording = false;
  int _recordSecondsRemaining = 5;
  Timer? _recordUiTimer;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller.initialize().then((_) {
      if (mounted) {
        _controller.startImageStream(_processCameraImage);
      }
    });
  }

  void _processCameraImage(CameraImage cameraImage) {
    if (mounted) {
      context.read<SegmentationCubit>().processCameraImage(cameraImage);
    }
  }

  void _submitTextPrompt() {
    final text = _promptController.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enter an object name to segment (e.g. "coffee mug").'),
        ),
      );
      return;
    }
    FocusScope.of(context).unfocus();
    context.read<SegmentationCubit>().setPrototypeFromGemma(
      gemmaService: widget.inferenceService,
      textTarget: text,
    );
  }

  Future<void> _toggleVoiceEnrollment() async {
    final cubit = context.read<SegmentationCubit>();

    // If already recording, stop early and trigger Gemma 4 immediately
    if (_isRecording) {
      _recordUiTimer?.cancel();
      await _audioService.stopRecording((path) {
        if (!mounted) return;
        setState(() => _isRecording = false);
        if (path != null) {
          cubit.setPrototypeFromGemma(
            gemmaService: widget.inferenceService,
            audioPath: path,
          );
        }
      });
      return;
    }

    setState(() {
      _isRecording = true;
      _recordSecondsRemaining = 5;
    });

    _recordUiTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() {
        if (_recordSecondsRemaining > 1) {
          _recordSecondsRemaining--;
        } else {
          timer.cancel();
        }
      });
    });

    await _audioService.startRecordingWithLimit(
      timeLimitSeconds: 5,
      onRecordingComplete: (path) {
        _recordUiTimer?.cancel();
        if (!mounted) return;
        setState(() => _isRecording = false);

        if (path != null) {
          cubit.setPrototypeFromGemma(
            gemmaService: widget.inferenceService,
            audioPath: path,
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                'Microphone permission denied or recording failed.',
              ),
              backgroundColor: Colors.red,
            ),
          );
        }
      },
    );
  }

  @override
  void dispose() {
    _recordUiTimer?.cancel();
    _audioService.dispose();
    _promptController.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<SegmentationCubit, SegmentationState>(
      listenWhen: (previous, current) => previous.status != current.status,
      listener: (context, state) {
        if (state.status == SegmentationStatus.error &&
            state.errorMessage != null) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('❌ ${state.errorMessage}'),
              backgroundColor: Colors.red,
            ),
          );
        } else if (state.status == SegmentationStatus.prototypeReady) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('✅ Prototype created! Tracking active.'),
              duration: Duration(seconds: 2),
            ),
          );
        }
      },
      builder: (context, state) {
        final cubit = context.read<SegmentationCubit>();
        final bool isPrototypeReady =
            state.status == SegmentationStatus.prototypeReady;
        final bool isCreatingPrototype =
            state.status == SegmentationStatus.creatingPrototype;

        return Scaffold(
          appBar: AppBar(
            title: const Text('Gemma 4 + DINOv3 Tracker'),
            actions: [
              PopupMenuButton<int>(
                initialValue: state.selectedInputSize,
                tooltip: 'Select DINOv3 Input Size',
                onSelected: cubit.updateInputSize,
                itemBuilder: (BuildContext context) {
                  return [320, 400, 512, 768].map((int size) {
                    return PopupMenuItem<int>(
                      value: size,
                      child: Text('Input Size: $size'),
                    );
                  }).toList();
                },
                icon: const Icon(Icons.aspect_ratio),
              ),
              IconButton(
                icon: Icon(
                  Icons.tune,
                  color: state.showSlider
                      ? Theme.of(context).colorScheme.secondary
                      : null,
                ),
                tooltip: 'Toggle Threshold Slider',
                onPressed: cubit.toggleSliderVisibility,
              ),
            ],
          ),
          body: FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    // Live Camera Feed
                    CameraPreview(_controller),

                    // Real-time DINOv3 Patch Similarity Mask Overlay
                    if (state.overlayImage != null)
                      FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: state.overlayImage!.width.toDouble(),
                          height: state.overlayImage!.height.toDouble(),
                          child: CustomPaint(
                            painter: OverlayPainter(state.overlayImage!),
                          ),
                        ),
                      ),

                    // Gemma 4 Grounding Bounding Box & Core Seed Box Overlay
                    if (state.detectedBox != null && !state.isSegmenting)
                      CustomPaint(
                        painter: BoundingBoxPainter(state.detectedBox!),
                      ),

                    // Top Text & Voice Enrollment Bar
                    Positioned(
                      top: 12,
                      left: 12,
                      right: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.65),
                          borderRadius: BorderRadius.circular(28),
                        ),
                        child: Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: _promptController,
                                enabled: !isCreatingPrototype && !_isRecording,
                                style: const TextStyle(color: Colors.white),
                                textInputAction: TextInputAction.search,
                                onSubmitted: (_) => _submitTextPrompt(),
                                decoration: const InputDecoration(
                                  hintText: 'Segment object (or tap mic)...',
                                  hintStyle: TextStyle(color: Colors.white60),
                                  border: InputBorder.none,
                                  isDense: true,
                                ),
                              ),
                            ),
                            IconButton(
                              onPressed: isCreatingPrototype || _isRecording
                                  ? null
                                  : _submitTextPrompt,
                              icon: const Icon(
                                Icons.send,
                                color: Colors.lightBlueAccent,
                              ),
                              tooltip: 'Find with Gemma 4',
                            ),
                            IconButton(
                              onPressed: isCreatingPrototype
                                  ? null
                                  : _toggleVoiceEnrollment,
                              icon: Icon(
                                _isRecording ? Icons.stop_circle : Icons.mic,
                                color: _isRecording
                                    ? Colors.redAccent
                                    : Colors.white,
                              ),
                              tooltip: _isRecording
                                  ? 'Stop Recording ($_recordSecondsRemaining s)'
                                  : 'Voice Command (5s)',
                            ),
                          ],
                        ),
                      ),
                    ),

                    // Status Banner during Recording or Gemma 4 Grounding
                    if (_isRecording || isCreatingPrototype)
                      Center(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 14,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black87,
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const SizedBox(
                                width: 22,
                                height: 22,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2.5,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 14),
                              Text(
                                _isRecording
                                    ? 'Listening... ($_recordSecondsRemaining s) Tap mic to finish'
                                    : 'Gemma 4 locating target & building prototype...',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                    // Cosine Similarity Threshold Slider
                    if (state.showSlider)
                      Positioned(
                        bottom: 120,
                        left: 20,
                        right: 88,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.6),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.tune, color: Colors.white),
                              Expanded(
                                child: Slider(
                                  value: state.similarityThreshold,
                                  min: 0.5,
                                  max: 0.9,
                                  divisions: 8,
                                  label: state.similarityThreshold
                                      .toStringAsFixed(2),
                                  onChanged: cubit.updateThreshold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                  ],
                );
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
          floatingActionButton: Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              // Fallback: Upload RGBA PNG from Gallery
              FloatingActionButton(
                heroTag: 'btn_upload_png',
                onPressed: isCreatingPrototype
                    ? null
                    : cubit.setReferencePrototype,
                tooltip: 'Set RGBA PNG Reference from Gallery',
                child: const Icon(Icons.upload_file),
              ),
              const SizedBox(height: 16),
              // Start / Stop Live DINOv3 Segmentation
              FloatingActionButton(
                heroTag: 'btn_toggle_seg',
                onPressed: isCreatingPrototype
                    ? null
                    : () {
                        if (!isPrototypeReady) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text(
                                'Use voice, text, or gallery upload to set a reference first!',
                              ),
                            ),
                          );
                          return;
                        }
                        cubit.toggleSegmentation();
                      },
                tooltip: 'Toggle Segmentation',
                backgroundColor: isCreatingPrototype ? Colors.grey : null,
                child: Icon(state.isSegmenting ? Icons.stop : Icons.play_arrow),
              ),
              const SizedBox(height: 16),
              // Toggle Largest Connected Component Filtering
              FloatingActionButton(
                heroTag: 'btn_largest_area',
                onPressed: cubit.toggleLargestAreaOnly,
                tooltip: 'Toggle Largest Area Only',
                backgroundColor: state.showLargestAreaOnly
                    ? Theme.of(context).primaryColor
                    : Colors.grey,
                child: const Icon(Icons.filter_center_focus),
              ),
              if (state.cutoutPngPath != null) ...[
                FloatingActionButton(
                  heroTag: 'btn_save_cutout',
                  onPressed: (isCreatingPrototype || state.isSavingCutout)
                      ? null
                      : () async {
                          final saved = await cubit.exportCutoutToGallery();
                          if (context.mounted && saved) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text(
                                  '💾 Transparent RGBA PNG saved to Gallery!',
                                ),
                                duration: Duration(seconds: 2),
                              ),
                            );
                          }
                        },
                  tooltip: 'Export EdgeTAM Cutout as RGBA PNG',
                  backgroundColor: Colors.teal,
                  child: state.isSavingCutout
                      ? const SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.save_alt, color: Colors.white),
                ),
                const SizedBox(height: 16),
              ],
            ],
          ),
        );
      },
    );
  }
}

class OverlayPainter extends CustomPainter {
  final ui.Image image;
  OverlayPainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: image,
      fit: BoxFit.fill,
    );
  }

  @override
  bool shouldRepaint(covariant OverlayPainter oldDelegate) =>
      oldDelegate.image != image;
}

/// Draws Gemma 4's detected bounding box (cyan) and the inner 36% DINOv3 seed core (amber)
class BoundingBoxPainter extends CustomPainter {
  final List<double> box; // [ymin, xmin, ymax, xmax] normalized 0.0..1.0
  BoundingBoxPainter(this.box);

  @override
  void paint(Canvas canvas, Size size) {
    if (box.length < 4) return;

    final ymin = box[0] * size.height;
    final xmin = box[1] * size.width;
    final ymax = box[2] * size.height;
    final xmax = box[3] * size.width;

    final outerRect = Rect.fromLTRB(xmin, ymin, xmax, ymax);
    final outerPaint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    canvas.drawRect(outerRect, outerPaint);

    // Also visualize the inner 36% core used as the DINOv3 Center-Seed
    final marginX = (xmax - xmin) * 0.32;
    final marginY = (ymax - ymin) * 0.32;
    final coreRect = Rect.fromLTRB(
      xmin + marginX,
      ymin + marginY,
      xmax - marginX,
      ymax - marginY,
    );
    final corePaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(coreRect, corePaint);
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) =>
      oldDelegate.box != box;
}
