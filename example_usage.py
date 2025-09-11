#!/usr/bin/env python3
"""
Quick start example for the voice recognition system.
This script demonstrates the basic workflow without requiring audio hardware.
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_example_workflow():
    """Create an example workflow description."""
    
    print("Voice Recognition System - Quick Start Example")
    print("=" * 60)
    
    print("\n📋 Typical Workflow:")
    print("1. Record training data:")
    print("   python src/sample_recorder.py")
    print("   → Records voice samples from 4 speakers")
    print("   → Creates run_YYYYMMDD_HHMMSS/ directory")
    print("   → Generates visualizations and saves metadata")
    
    print("\n2. Test voice classification:")
    print("   python src/main.py run_YYYYMMDD_HHMMSS")
    print("   → Loads trained model")
    print("   → Records test samples or classifies existing ones")
    print("   → Saves predictions to CSV")
    
    print("\n📊 Expected Output Structure:")
    example_dir = "run_20231215_143022"
    files = [
        "train_p1_t1.wav through train_p4_t12.wav (48 training files)",
        "persona_1_tomas_time.png (time domain plots)",
        "david_fft.png, gabo_fft.png, dante_fft.png, gal_fft.png",
        "meta.npz (training metadata)",
        "test_1.wav through test_6.wav (test recordings)",
        "test_1_fft.png through test_6_fft.png (test analysis)",
        "predictions.csv (classification results)"
    ]
    
    print(f"\n{example_dir}/")
    for file in files:
        print(f"  ├── {file}")
    
    print("\n🎯 Configuration Options:")
    print("Edit these variables in the source files to customize:")
    
    config_options = [
        ("N_PERSONS", "4", "Number of speakers to train"),
        ("N_TAKES_TRAIN", "12", "Training samples per speaker"),
        ("PERSON_NAMES", '["David", "Gabo", "Dante", "Gal"]', "Speaker names"),
        ("TAKE_SECONDS", "3.0", "Recording duration per sample"),
        ("EXPECTED_SR", "16000", "Audio sampling rate"),
        ("N_TESTS", "6", "Number of test samples")
    ]
    
    for param, default, description in config_options:
        print(f"  {param:<15} = {default:<10} # {description}")
    
    print("\n🔬 Technical Details:")
    print("  • Feature extraction: 32 frequency bands (18 low + 14 high)")
    print("  • Classification: Centroid-based with Euclidean distance")
    print("  • Audio format: 16-bit mono WAV at 16kHz")
    print("  • Preprocessing: RMS normalization + Z-score standardization")
    
    print("\n📈 Performance Tips:")
    tips = [
        "Record in a quiet environment",
        "Maintain consistent microphone distance",
        "Speak clearly and naturally",
        "Ensure speakers have distinct vocal characteristics",
        "Use a good quality microphone"
    ]
    
    for tip in tips:
        print(f"  • {tip}")

def show_sample_predictions():
    """Show example prediction output format."""
    
    print("\n📊 Example Prediction Output:")
    print("-" * 40)
    
    # Simulate some example predictions
    examples = [
        ("test_1.wav", "David", [0.234, 0.891, 1.456, 2.123]),
        ("test_2.wav", "Gabo", [1.445, 0.156, 1.234, 1.987]),
        ("test_3.wav", "Dante", [2.001, 1.789, 0.198, 1.456]),
        ("test_4.wav", "Gal", [1.789, 1.234, 1.567, 0.145])
    ]
    
    for filename, prediction, distances in examples:
        dist_str = ', '.join(f'{d:.3f}' for d in distances)
        print(f"[pred] {filename} -> {prediction}  (distances: {dist_str})")
    
    print("\nCSV Output (predictions.csv):")
    print("prueba,prediccion")
    for i, (_, prediction, _) in enumerate(examples, 1):
        print(f"{i},{prediction}")

def main():
    """Main example function."""
    
    # Check if we're in the correct directory
    if not os.path.exists("src/sample_recorder.py"):
        print("⚠ This script should be run from the project root directory")
        print("  (where src/ folder is located)")
        return
    
    create_example_workflow()
    show_sample_predictions()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to get started!")
    print("Run 'python check_dependencies.py' first to verify your setup.")

if __name__ == "__main__":
    main()