
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Error initializing cameras: $e');
  }
  runApp(const PlantDiseaseApp());
}

class PlantDiseaseApp extends StatelessWidget {
  const PlantDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Clinic',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
      ),
      home: const DiseaseDetectionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DiseaseDetectionScreen extends StatefulWidget {
  const DiseaseDetectionScreen({super.key});

  @override
  State<DiseaseDetectionScreen> createState() => _DiseaseDetectionScreenState();
}

class _DiseaseDetectionScreenState extends State<DiseaseDetectionScreen> {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  List<String>? _labels;
  bool _isProcessing = false;
  String _resultText = "Point camera at a leaf and press Scan";
  String _cureText = "";

  // Hardcoded Cure/Treatment Database for your placement showcase
  final Map<String, String> _treatmentDatabase = {
    "daisy": "Ensure optimal sunlight (6+ hours). Water when top 1 inch of soil is dry. Watch for aphids.",
    "dandelion": "Typically classified as a weed. If cultivating, ensure well-drained soil. Regular pruning prevents overgrowth.",
    "roses": "Prune dead stems in early spring. Treat fungal black spots with organic neem oil spray. Water at the base.",
    "sunflowers": "Requires deep watering during flowering stage. Support tall stems with stakes. Protect leaves from birds.",
    "tulips": "Keep soil moist but not soggy. Remove spent blooms immediately. Store bulbs in a cool, dry place after foliage dies.",
  };

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) return;
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
    await _cameraController!.initialize();
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/plant_disease_model.tflite');
      final labelString = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelString.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
      print("Model and labels loaded successfully!");
    } catch (e) {
      print("Failed to load model or labels: $e");
    }
  }

  Future<void> _runInference() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized || _interpreter == null || _labels == null || _isProcessing) {
      return;
    }

    setState(() {
      _isProcessing = true;
      _resultText = "Analyzing plant health...";
      _cureText = "";
    });

    try {
      // 1. Capture image
      final XFile file = await _cameraController!.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final originalImage = img.decodeImage(bytes);
      
      if (originalImage == null) return;

      // 2. Preprocess image to 224x224 (Matches MobileNetV2 input size from Phase 1)
      final resizedImage = img.copyResize(originalImage, width: 224, height: 224);
      
      // 3. Convert image pixels to Float32 array [1, 224, 224, 3]
      var input = List.generate(1, (_) => List.generate(224, (_) => List.generate(224, (_) => List.filled(3, 0.0))));
      
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resizedImage.getPixel(x, y);
          // Normalize pixel values to [-1, 1] range matching MobileNetV2 preprocessing
          input[0][y][x][0] = (pixel.r / 127.5) - 1.0;
          input[0][y][x][1] = (pixel.g / 127.5) - 1.0;
          input[0][y][x][2] = (pixel.b / 127.5) - 1.0;
        }
      }

      // 4. Prepare output buffer
      var output = List.filled(1 * _labels!.length, 0.0).reshape([1, _labels!.length]);

      // 5. Run prediction
      _interpreter!.run(input, output);

      // 6. Extract highest probability index
      List<double> probabilities = List<double>.from(output[0]);
      int maxIndex = 0;
      double maxProb = -1.0;
      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      String detectedClass = _labels![maxIndex];
      String treatment = _treatmentDatabase[detectedClass.toLowerCase()] ?? "No specific treatment details found for this variety. General recommendation: Maintain regular watering schedules and monitor for pests.";

      setState(() {
        _resultText = "${detectedClass.toUpperCase()} (${(maxProb * 100).toStringAsFixed(1)}%)";
        _cureText = treatment;
      });

    } catch (e) {
      setState(() {
        _resultText = "Error running diagnostic";
        _cureText = e.toString();
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Disease Detector', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Camera view panel
          Expanded(
            flex: 3,
            child: ClipRRect(
              child: CameraPreview(_cameraController!),
            ),
          ),
          
          // Result Information Panel
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
                boxShadow: [
                  BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 10, offset: const Offset(0, -5))
                ],
              ),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("DIAGNOSIS RESULT", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey)),
                    const SizedBox(height: 4),
                    Text(_resultText, style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.green.shade900)),
                    const Divider(height: 24),
                    const Text("RECOMMENDED CURE / TREATMENT", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey)),
                    const SizedBox(height: 4),
                    Text(
                      _cureText.isEmpty ? "Results will update immediately after a successful scan." : _cureText,
                      style: const TextStyle(fontSize: 15, color: Colors.black87, height: 1.4),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isProcessing ? null : _runInference,
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
        icon: _isProcessing 
          ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5))
          : const Icon(Icons.camera_alt),
        label: Text(_isProcessing ? "Processing..." : "Scan Crop Layer"),
      ),
    );
  }
}