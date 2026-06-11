#!/usr/bin/env python3
"""
Model Converter: H5 to TensorFlow Lite
Converts the plant disease detection model to TFLite format for mobile deployment
"""

import tensorflow as tf
import os

def convert_h5_to_tflite():
    """Convert H5 model to TensorFlow Lite format"""
    
    # Input and output paths
    h5_model_path = "plant_disease_model.h5"
    tflite_model_path = "android/app/src/main/assets/plant_disease_model.tflite"
    
    # Check if input model exists
    if not os.path.exists(h5_model_path):
        print(f"ERROR: Model file not found at {h5_model_path}")
        return False
    
    print("Loading H5 model...")
    try:
        model = tf.keras.models.load_model(h5_model_path)
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    print("\nConverting to TensorFlow Lite...")
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization for mobile (optional, comment out for faster conversion)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Create assets directory if it doesn't exist
        assets_dir = os.path.dirname(tflite_model_path)
        os.makedirs(assets_dir, exist_ok=True)
        
        # Save the converted model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ Conversion successful!")
        print(f"✓ TFLite model saved to: {tflite_model_path}")
        
        # Get file sizes
        h5_size = os.path.getsize(h5_model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
        
        print(f"\nModel Statistics:")
        print(f"  Original H5 size: {h5_size:.2f} MB")
        print(f"  TFLite size: {tflite_size:.2f} MB")
        print(f"  Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Plant Disease Model Converter")
    print("=" * 60)
    print()
    
    success = convert_h5_to_tflite()
    
    if success:
        print("\n✓ All done! You can now build the Android app.")
        print("\nNext steps:")
        print("1. Create labels.txt file with your disease class names")
        print("2. Build the Android app: cd android && ./gradlew assembleDebug")
    else:
        print("\n✗ Conversion failed. Please check the errors above.")
    
    print()
