import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ImageClassification(),
    );
  }
}

class ImageClassification extends StatefulWidget {
  @override
  _ImageClassificationState createState() => _ImageClassificationState();
}

class _ImageClassificationState extends State<ImageClassification> {
  late tfl.Interpreter _interpreter;
  String _result = '';
  String _ocrResult = '';
  List<String> _labels = ["AADHAAR", "PAN", "NONE"];
  final ImagePicker _picker = ImagePicker();
  File? _image;
  List<double> _probabilities = [];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/model_unquant.tflite');
      print('Model loaded successfully');
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  Future<void> classifyImage(File image) async {
    // Load and process the image
    final inputImage = img.decodeImage(image.readAsBytesSync())!;
    final processedImage = processImage(inputImage);

    // Define the output buffer
    var output = List.filled(3, 0.0).reshape([1, 3]);

    // Run the model on the input
    _interpreter.run(processedImage, output);

    // Process the output
    setState(() {
      _probabilities = output[0];
      _result = 'Classification Results:\n';
      for (int i = 0; i < _labels.length; i++) {
        _result += '${_labels[i]}: ${(_probabilities[i] * 100).toStringAsFixed(2)}%\n';
      }
    });

    // Perform OCR on the image
    performOCR(image);
  }

  Uint8List processImage(img.Image image) {
    // Resize the image to 224x224 as required by the model
    img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

    // Create a byte buffer to hold the image data
    var byteBuffer = Float32List(1 * 224 * 224 * 3);
    var buffer = Float32List.view(byteBuffer.buffer);
    int bufferIndex = 0;

    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        var pixel = resizedImage.getPixel(x, y);
        buffer[bufferIndex++] = (img.getLuminance(pixel) - 127.5) / 127.5; // Red
        buffer[bufferIndex++] = (img.getLuminance(pixel) - 127.5) / 127.5; // Green
        buffer[bufferIndex++] = (img.getLuminance(pixel) - 127.5) / 127.5; // Blue
      }
    }
    return byteBuffer.buffer.asUint8List();
  }

  Future<void> performOCR(File image) async {
    final inputImage = InputImage.fromFile(image);
    final textRecognizer = TextRecognizer();
    final recognizedText = await textRecognizer.processImage(inputImage);

    setState(() {
      _ocrResult = 'OCR Results:\n';
      for (var block in recognizedText.blocks) {
        for (var line in block.lines) {
          _ocrResult += '${line.text}\n';
        }
      }
    });

    textRecognizer.close();
  }

  Future<void> pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _image = File(image.path);
      });
      classifyImage(File(image.path));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Classification'),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _image != null
                  ? Container(
                width: 300,
                height: 300,
                child: Image.file(_image!, fit: BoxFit.cover),
              )
                  : Container(),
              SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () => pickImage(ImageSource.gallery),
                    child: Text('Pick Image'),
                  ),
                  SizedBox(width: 20),
                  ElevatedButton(
                    onPressed: () => pickImage(ImageSource.camera),
                    child: Text('Capture Image'),
                  ),
                ],
              ),
              SizedBox(height: 20),
              Text(_result),
              SizedBox(height: 20),
              Text(_ocrResult),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }
}
