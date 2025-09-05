# Signals Challenge

A comprehensive signal processing project featuring real-time frequency detection, wireless audio streaming, and voice identification using ESP32, Python, and MATLAB.

## 🎯 Project Overview

This project combines **real-time signal processing**, **wireless communication**, and **machine learning** to create a complete audio analysis system with two main components:

1. **Real-time Frequency Detection System** - ESP32 captures audio and streams it wirelessly to a Python server for real-time frequency analysis
2. **Voice Identification System** - MATLAB-based speaker recognition using feature extraction and centroid classification

## ✨ Features

### Real-time Audio Processing
- **High-quality audio capture** using ESP32-S3 with I2S PDM microphone
- **Advanced DSP pipeline** with filtering, noise gating, and soft limiting
- **Secure wireless streaming** with HMAC-SHA256 authentication
- **Real-time frequency detection** using FFT with parabolic interpolation
- **Sub-bin frequency resolution** for precise tone detection

### Voice Identification
- **Multi-speaker voice recording** and dataset creation
- **Feature extraction** (log energy, zero-crossing rate, spectral features)
- **Centroid-based classification** for speaker identification
- **Real-time voice recognition** with confidence scoring

### Signal Processing Techniques
- Fast Fourier Transform (FFT) analysis
- Parabolic interpolation for sub-bin resolution
- High-pass and low-pass filtering
- Noise gate with attack/release parameters
- Soft clipping limiter
- Spectral centroid and bandwidth analysis

## 🛠️ Hardware Requirements

### ESP32 Setup
- **ESP32-S3** development board
- **I2S PDM microphone** (connected to pins 41/42)
- **WiFi connection** (2.4 GHz)
- **Power supply** (USB or external)

### Development Environment
- **Python 3.7+** with NumPy
- **Arduino IDE** with ESP32 board support
- **MATLAB** R2018b or newer

## 📦 Software Dependencies

### Python Requirements
```bash
pip install numpy
```

### Arduino Libraries
- ESP_I2S
- WiFi
- mbedtls (for HMAC)

### MATLAB Toolboxes
- Signal Processing Toolbox
- Audio Toolbox

## 🚀 Quick Start

### 1. ESP32 Audio Streaming Setup

1. **Configure WiFi credentials** in `src/audio_for_signals.ino`:
   ```cpp
   const char* SSID = "YOUR_WIFI_SSID";
   const char* PASS = "YOUR_WIFI_PASSWORD";
   ```

2. **Set server IP** in the Arduino code:
   ```cpp
   #define AUDIO_SERVER_IP "192.168.1.100"  // Your Python server IP
   ```

3. **Upload the sketch** to your ESP32-S3

4. **Connect I2S PDM microphone**:
   - Clock pin: GPIO 42
   - Data pin: GPIO 41

### 2. Python Frequency Detection Server

1. **Update server configuration** in `src/main.py`:
   ```python
   HOST = "0.0.0.0"  # Listen on all interfaces
   PORT = 12345
   ```

2. **Create the missing actions.py file**:
   ```python
   # src/actions.py
   def route_frequency(freq_hz: float, confidence_db: float) -> bool:
       """
       Route detected frequencies to specific actions.
       
       Args:
           freq_hz: Detected frequency in Hz
           confidence_db: Detection confidence in dB
           
       Returns:
           bool: True if action was triggered, False otherwise
       """
       # Example: Trigger action for tones between 440-880 Hz with high confidence
       if 440 <= freq_hz <= 880 and confidence_db > 10:
           print(f"🎵 Musical note detected: {freq_hz:.1f} Hz")
           return True
       return False
   ```

3. **Run the server**:
   ```bash
   cd src
   python main.py
   ```

### 3. Voice Identification (MATLAB)

1. **Record voice samples**:
   ```matlab
   cd Matlab
   run('recopilacion_datos.m')
   ```
   - Follow prompts to record 3 voice samples per person
   - Data is saved as `datos_YOURNAME.mat`

2. **Train and test voice recognition**:
   ```matlab
   run('Identificacion_voz.m')
   ```
   - Loads all voice data files
   - Trains centroid-based classifier
   - Records new sample and predicts speaker

## 📁 Project Structure

```
signals_challenge/
├── src/
│   ├── main.py              # Python frequency detection server
│   ├── audio_for_signals.ino # ESP32 audio capture and streaming
│   └── actions.py           # Frequency routing logic (create this file)
├── Matlab/
│   ├── recopilacion_datos.m   # Voice data collection script
│   ├── Identificacion_voz.m   # Voice identification script
│   ├── datos_David.mat        # Voice data for David
│   ├── datos_Gal.mat         # Voice data for Gal
│   └── datos_gabo.mat        # Voice data for gabo
└── README.md
```

## 🔧 Configuration

### Audio Parameters
- **Sample Rate**: 16 kHz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Buffer Size**: 256 samples
- **Frame Rate**: ~62.5 Hz

### Network Security
- **Authentication**: HMAC-SHA256
- **Shared Key**: 256-bit hex key (configurable)
- **Protocol**: TCP with custom handshake

### DSP Parameters
```cpp
#define VOLUME_GAIN        3.0f     // Audio gain multiplier
#define GATE_ATTACK_MS     5        // Gate attack time
#define GATE_RELEASE_MS    60       // Gate release time
#define THRESH_OPEN        120.0f   // Gate open threshold
#define THRESH_CLOSE       60.0f    // Gate close threshold
```

## 🎛️ Usage Examples

### Frequency Detection
The system can detect and analyze:
- **Musical tones** and instruments
- **Voice frequencies** and formants
- **Environmental sounds** and signals
- **Ultrasonic frequencies** (up to 8 kHz)

### Voice Recognition Applications
- **Smart home control** with voice commands
- **Security systems** with speaker verification
- **Audio logging** with automatic speaker tagging
- **Educational tools** for voice analysis

## 🔬 Technical Details

### Frequency Detection Algorithm
1. **Audio preprocessing** with Hann windowing
2. **Zero-padding** to next power of 2
3. **Real FFT** computation
4. **Peak detection** in magnitude spectrum
5. **Parabolic interpolation** for sub-bin accuracy
6. **Confidence estimation** using signal-to-noise ratio

### Voice Recognition Pipeline
1. **Feature extraction**:
   - Log energy
   - Zero-crossing rate
   - Spectral centroid
   - Spectral bandwidth
2. **Training**: Compute feature centroids per speaker
3. **Classification**: Nearest centroid using Euclidean distance

### Communication Protocol
1. **TCP connection** establishment
2. **HMAC authentication** with device MAC address
3. **WAV header** transmission
4. **Continuous audio streaming** in 256-sample chunks

## 🐛 Troubleshooting

### Common Issues

**ESP32 won't connect to WiFi:**
- Check SSID/password configuration
- Ensure 2.4 GHz network (5 GHz not supported)
- Verify signal strength and range

**Python server connection fails:**
- Check firewall settings
- Verify IP address configuration
- Ensure port 12345 is available

**Audio quality issues:**
- Check microphone connections
- Adjust DSP parameters (gain, thresholds)
- Verify I2S pin configuration

**MATLAB recording problems:**
- Check audio device permissions
- Verify Audio Toolbox installation
- Adjust recording parameters if needed

## 🎓 Educational Applications

This project demonstrates key concepts in:
- **Digital Signal Processing** (FFT, filtering, windowing)
- **Embedded Systems** (ESP32, I2S, real-time processing)
- **Network Programming** (TCP, authentication, protocols)
- **Machine Learning** (feature extraction, classification)
- **Audio Processing** (frequency analysis, voice recognition)

## 🔮 Future Enhancements

- **Multi-channel audio** support
- **Advanced ML models** (neural networks)
- **Real-time visualization** web interface
- **Mobile app** integration
- **Cloud-based processing** and storage
- **Multiple ESP32 nodes** for distributed sensing

## 📄 License

This project is open source. Feel free to modify and extend for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional signal processing algorithms
- Enhanced voice recognition models
- Web-based user interface
- Documentation and examples
- Hardware integration guides

---

*Built with ❤️ for signal processing education and research*