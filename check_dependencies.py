#!/usr/bin/env python3
"""
Dependency check script for the voice recognition system.
Run this to verify all required packages are installed.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are available."""
    
    dependencies = [
        ("numpy", "Scientific computing"),
        ("matplotlib", "Plotting and visualization"),
        ("dotenv", "Environment variable loading"),
    ]
    
    audio_dependencies = [
        ("sounddevice", "Audio recording and playback"),
    ]
    
    print("Checking core dependencies...")
    missing_core = []
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✓ {package:<15} - {description}")
        except ImportError:
            print(f"✗ {package:<15} - {description} (MISSING)")
            missing_core.append(package)
    
    print("\nChecking audio dependencies...")
    missing_audio = []
    
    for package, description in audio_dependencies:
        try:
            __import__(package)
            print(f"✓ {package:<15} - {description}")
        except ImportError:
            print(f"✗ {package:<15} - {description} (MISSING)")
            missing_audio.append(package)
        except OSError as e:
            if "PortAudio" in str(e):
                print(f"⚠ {package:<15} - {description} (PortAudio system library needed)")
                print("    Install PortAudio: Ubuntu/Debian: sudo apt-get install portaudio19-dev")
                print("                       macOS: brew install portaudio")
            else:
                print(f"✗ {package:<15} - {description} (ERROR: {e})")
                missing_audio.append(package)
    
    print("\n" + "="*60)
    
    if not missing_core and not missing_audio:
        print("✓ All dependencies are installed!")
        print("The voice recognition system should work correctly.")
        return True
    else:
        if missing_core:
            print(f"✗ Missing core dependencies: {', '.join(missing_core)}")
            print("Install with: pip install " + " ".join(missing_core))
        
        if missing_audio:
            print(f"⚠ Missing audio dependencies: {', '.join(missing_audio)}")
            print("Install with: pip install " + " ".join(missing_audio))
            print("Note: sounddevice also requires PortAudio system library.")
            print("Ubuntu/Debian: sudo apt-get install portaudio19-dev")
            print("macOS: brew install portaudio")
        
        return False

def test_audio_system():
    """Test if audio system is working (optional)."""
    try:
        import sounddevice as sd
        print("\nTesting audio system...")
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices:")
        
        # Find default input device
        default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
        if default_input is not None:
            device_info = sd.query_devices(default_input)
            print(f"Default input: {device_info['name']}")
            print("✓ Audio system appears to be working")
        else:
            print("⚠ No default input device found")
            
    except Exception as e:
        print(f"⚠ Audio system test failed: {e}")
        print("This may indicate PortAudio is not properly installed.")

if __name__ == "__main__":
    print("Voice Recognition System - Dependency Check")
    print("="*60)
    
    success = check_dependencies()
    
    if success:
        test_audio_system()
    
    print("\n" + "="*60)
    if success:
        print("Ready to use! Try running:")
        print("  python src/sample_recorder.py  # To record training data")
        print("  python src/main.py <run_dir>  # To test classification")
    else:
        print("Please install missing dependencies before proceeding.")
        sys.exit(1)