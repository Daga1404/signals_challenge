# Voice Recognition System with ESP32-S3

A comprehensive voice identification system implemented in two ways: using an ESP32-S3 microcontroller with Python for machine learning-based voice recognition, and using Matlab for signal processing and classification. Both approaches allow recording training samples from multiple people and classifying new voice samples to identify the speaker.

## 🎯 Project Overview

This project implements a complete voice recognition pipeline consisting of:

- **ESP32-S3 Audio Capture**: Real-time audio streaming with built-in DSP processing
- **Voice Training**: Recording and processing voice samples from multiple speakers
- **Machine Learning Classification**: FFT-based feature extraction with centroid classification
- **Real-time Prediction**: Live voice classification via TCP streaming

## 🛠️ Hardware Requirements

### ESP32-S3 Setup
- **ESP32-S3** development board
- **I2S Microphone** (compatible with ESP32-S3 I2S interface)
- **WiFi Network** (2.4GHz) for TCP communication
- **Micro-USB Cable** for programming and power

### Microphone Connection
The system expects an I2S microphone connected to the ESP32-S3. Ensure proper I2S pin configuration in your hardware setup.

## 📦 Software Requirements

### Arduino Environment
- **Arduino IDE** or **PlatformIO**
- **ESP32 Board Package** (ESP32-S3 support)
- Required libraries:
  - `WiFi` (built-in)
  - `ESP_I2S` 
  - `mbedtls` (for HMAC-SHA256)

### Python Environment
- **Python 3.7+**
- Required packages:
```bash
pip install numpy matplotlib python-dotenv
```

**Note**: `wave`, `socket`, `hmac`, and other modules are part of Python's standard library.

## ⚙️ Configuration

### 1. Create Configuration File
Create a `config.h` file in the Arduino project directory:

```cpp
// config.h
#ifndef CONFIG_H
#define CONFIG_H

// WiFi Configuration
#define WIFI_SSID     "YourWiFiNetwork"
#define WIFI_PASSWORD "YourWiFiPassword"

// Network Configuration  
#define TCP_PORT      8888
#define SHARED_KEY_HEX "your_64_character_hex_key_here"

// I2S Pin Configuration (adjust for your hardware)
#define I2S_WS    42  // Word Select (LRCLK)
#define I2S_SD    41  // Serial Data 
#define I2S_SCK   40  // Serial Clock (BCLK)

#endif
```

### 2. Create Environment File
Create a `.env` file in the project root:

```bash
# .env
ESP_HOST=192.168.1.100  # IP address of your computer (not ESP32)
ESP_PORT=8888
ESP_SHARED_KEY_HEX=your_64_character_hex_key_here
```

**Finding Your Computer's IP**: 
- Windows: `ipconfig`
- Linux/macOS: `ip addr` or `ifconfig`
- The ESP32 will connect TO this address (your computer runs the server)

**Note**: The `SHARED_KEY_HEX` must be the same in both `config.h` and `.env` files.

### Generating a Secure Key
To generate a secure 64-character hex key:
```bash
# Linux/macOS
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_hex(32))"
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd signals_challenge
   ```

2. **Install Python dependencies**
   ```bash
   pip install numpy matplotlib python-dotenv
   ```

3. **Configure the system** (create `config.h` and `.env` as described above)

4. **Upload ESP32 firmware** (using Arduino IDE)

5. **Record training data**
   ```bash
   cd src
   python sample_recorder.py
   ```

6. **Test voice classification**
   ```bash
   python main.py run_YYYYMMDD_HHMMSS/
   ```

## 📋 Detailed Usage

### Phase 1: Training Data Collection

1. **Upload Arduino Code**
   ```bash
   # Open audio_for_signals.ino in Arduino IDE
   # Select Board: "ESP32S3 Dev Module" 
   # Select Port: (your ESP32-S3 port)
   # Upload the sketch
   ```
   
   **Important**: Make sure to select the correct ESP32-S3 board variant in Arduino IDE. The serial monitor will show WiFi connection status and audio system initialization.

2. **Record Training Samples**
   ```bash
   cd src
   python sample_recorder.py
   ```
   
   The script will:
   - Wait for ESP32 connection
   - Guide you through recording 5 samples per person (configurable)
   - Save WAV files as `train_p{person}_t{take}.wav`
   - Generate time-domain and frequency plots
   - Create metadata file `meta.npz`

### Phase 2: Voice Classification

3. **Run Classification**
   ```bash
   python main.py run_YYYYMMDD_HHMMSS/
   ```
   
   The system will:
   - Load training data and build the model
   - Look for existing test files or record new ones
   - Classify voice samples and show predictions
   - Save results to `predictions.csv`

## 📁 Project Structure

```
signals_challenge/
├── audio_for_signals.ino    # ESP32-S3 firmware
├── src/
│   ├── sample_recorder.py   # Training data collection
│   └── main.py             # Voice classification
├── config.h                # Arduino configuration (create this)
├── .env                    # Python environment variables (create this)
└── run_YYYYMMDD_HHMMSS/    # Generated training data folders
    ├── meta.npz            # Training metadata
    ├── train_p1_t1.wav     # Training samples
    ├── train_p1_t2.wav
    ├── ...
    ├── test_1.wav          # Test samples (optional)
    ├── predictions.csv     # Classification results
    └── *.png              # Generated plots
```

## 🧑‍🔬 Matlab Implementation

In addition to the Arduino+Python workflow, the project includes a full Matlab implementation for voice recognition. The Matlab approach allows you to:

- Record and process audio samples using Matlab scripts.
- Perform feature extraction, visualization, and classification directly in Matlab.
- Compare results and experiment with different DSP and ML techniques.

### Matlab Folder Contents

The `Matlab/` directory contains:
- `Identificacion_voz.m`: Main Matlab script for voice identification and analysis.
- `recopilacion_datos.m`: Script for collecting and organizing voice data in Matlab.
- `datos_David.mat`, `datos_gabo.mat`, `datos_Gal.mat`: Example Matlab data files containing recorded voice samples for different speakers.

You can use these scripts to:
- Explore and visualize audio signals in Matlab.
- Test alternative DSP or ML approaches.
- Compare Matlab results with the Python-based pipeline.

**Note:** The Matlab workflow is independent and does not require the ESP32 or Python scripts. It is useful for research, prototyping, and educational purposes, and provides a reference for signal processing and classification techniques.

## 🔧 Technical Details

### Audio Processing
- **Sample Rate**: 16 kHz
- **Format**: 16-bit mono PCM
- **Recording Length**: 3 seconds per sample
- **DSP Features**:
  - High-pass filter (DC removal)
  - Low-pass filter (anti-aliasing)
  - Noise gate with hysteresis
  - Soft clipping limiter

### Machine Learning
- **Feature Extraction**: FFT-based frequency bands (80 Hz - 5 kHz, 12 bands)
- **Preprocessing**: Z-score normalization
- **Classification**: Centroid-based distance classification
- **Model**: Simple but effective for voice identification

### Network Protocol
- **Transport**: TCP sockets
- **Authentication**: HMAC-SHA256 challenge-response
- **Audio Format**: WAV stream with 44-byte header
- **Security**: Shared key prevents unauthorized connections

## 🎛️ Configuration Parameters

### Default Settings (Configurable in Code)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_PERSONS` | 3 | Number of people to train |
| `N_TAKES_TRAIN` | 5 | Samples per person |
| `TAKE_SECONDS` | 3.0 | Recording length |
| `PERSON_NAMES` | ["david", "gal", "gabo"] | Speaker names |
| `SAMPLE_RATE` | 16000 | Audio sample rate |
| `N_TESTS` | 4 | Number of test samples |

### DSP Parameters (ESP32)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOLUME_GAIN` | 3.0 | Input gain multiplier |
| `THRESH_OPEN` | 120.0 | Noise gate open threshold |
| `THRESH_CLOSE` | 60.0 | Noise gate close threshold |
| `GATE_ATTACK_MS` | 5 | Gate attack time |
| `GATE_RELEASE_MS` | 60 | Gate release time |

## 🔍 Troubleshooting

### Common Issues

**ESP32 Won't Connect to WiFi**
- Verify SSID and password in `config.h`
- Ensure 2.4GHz network (ESP32 doesn't support 5GHz)
- Check signal strength and range
- Review serial output for connection details

**Authentication Failed**
- Verify `SHARED_KEY_HEX` matches in both files
- Ensure key is exactly 64 hexadecimal characters
- Check firewall settings on host computer

**Audio Quality Issues**
- Verify I2S microphone connections
- Check microphone power supply
- Adjust DSP parameters (gain, thresholds)
- Ensure proper grounding

**Python Connection Issues**
- Verify ESP32 IP address in `.env` file
- Check if port 8888 is available
- Disable firewall temporarily for testing
- Ensure both devices are on same network

**Poor Classification Accuracy**
- Record more training samples per person
- Ensure consistent recording conditions
- Check for background noise
- Verify microphone placement

### Debug Output

The system provides extensive logging:
- ESP32 serial output shows WiFi and audio status
- Python scripts show network, authentication, and processing steps
- Generated plots help visualize audio quality

## 🎨 Generated Outputs

The system creates several visualization files:
- **Time-domain plots**: Show raw audio waveforms for each speaker
- **Frequency-domain plots**: Display FFT analysis of voice characteristics
- **Test classification plots**: Visualize features used for prediction

## 📈 Performance Notes

- **Latency**: ~250ms warmup + recording time
- **Accuracy**: Depends on training data quality and speaker distinctiveness
- **Memory**: ESP32 uses ~256-sample audio buffers
- **Network**: Optimized for local network usage

## 🔮 Future Enhancements

Potential improvements:
- Support for more sophisticated ML models
- Real-time continuous recognition
- Multiple microphone array support
- Web interface for easy operation
- Mobile app integration

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

---

**Note**: This is a signals processing educational project. For production voice recognition systems, consider using more advanced ML frameworks and security measures.
