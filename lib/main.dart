import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'camera_screen.dart';
import 'inference_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));
}

Future<String> _getModelPath() async {
  final directory = await getExternalStorageDirectory();

  if (directory == null) {
    throw Exception('Could not access external storage.');
  }

  final modelFile = File('${directory.path}/model.litertlm');

  if (!await modelFile.exists()) {
    throw Exception(
      'Model file not found at ${modelFile.path}.\n\n'
      'Push your Gemma 4 model via ADB:\n'
      'adb push gemma-4-E2B-it.litertlm ${modelFile.path}',
    );
  }

  return modelFile.path;
}

class MyApp extends StatefulWidget {
  const MyApp({super.key, required this.camera});
  final CameraDescription camera;

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late Future<InferenceService> _initFuture;

  @override
  void initState() {
    super.initState();
    _initFuture = _initEngine();
  }

  Future<InferenceService> _initEngine() async {
    final modelPath = await _getModelPath();
    final inferenceService = InferenceService();

    // Yield briefly so Flutter can paint the loading spinner UI
    await Future.delayed(const Duration(milliseconds: 100));

    inferenceService.initializeEngine(modelPath);
    return inferenceService;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma 4 + DINOv3 Zero-Shot Tracker',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: FutureBuilder<InferenceService>(
        future: _initFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Scaffold(
              body: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 24),
                    Text(
                      'Initializing LiteRT-LM & DINOv3...',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Loading Gemma 4 weights into GPU memory',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ],
                ),
              ),
            );
          }

          if (snapshot.hasError) {
            return Scaffold(
              body: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Center(
                  child: SelectableText(
                    'Startup Error:\n${snapshot.error}',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.red, fontSize: 15),
                  ),
                ),
              ),
            );
          }

          return CameraScreen(
            camera: widget.camera,
            inferenceService: snapshot.data!,
          );
        },
      ),
    );
  }
}
