# PlantDiseaseApp
# 🌱 Crop Clinic: AI Plant Disease Detector

An intelligent, offline-first mobile application built with Flutter and TensorFlow Lite that diagnoses plant diseases in real-time and provides actionable treatment plans.

This project was originally conceptualized during a 12-hour hackathon and later re-architected into a production-ready Flutter application with on-device machine learning.

## ✨ Features

* **Real-Time Inference:** Uses the device camera to scan leaves and classify plant health instantly.
* **Edge AI (Offline ML):** Powered by a custom-trained TensorFlow Lite model, ensuring the app works flawlessly without an internet connection.
* **Treatment Database:** Automatically maps the detected disease to a curated database of actionable cures and care instructions.
* **Modern Cross-Platform UI:** Built with Flutter for smooth, native-like performance and a clean, accessible user interface.

## 🛠️ Tech Stack

* **Frontend:** Flutter & Dart
* **Machine Learning:** TensorFlow Lite (`tflite_flutter`)
* **Model Training:** Python, TensorFlow/Keras, Google Colab (MobileNetV2 Transfer Learning)
* **Hardware Integration:** `camera` plugin for native Android/iOS camera access

## 📂 Project Structure

```text
PlantDiseaseApp/
│
├── crop_app/                 # The main Flutter application codebase
│   ├── assets/models/        # Contains the compiled .tflite model and labels
│   ├── lib/                  # Dart UI and logic (main.dart)
│   └── pubspec.yaml          # Flutter dependencies
│
└── README.md                 # Project documentation
