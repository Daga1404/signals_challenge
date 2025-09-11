# Voice Recognition System

A Python-based voice recognition system that uses frequency domain analysis and machine learning to identify speakers. The system records audio samples, extracts spectral features, and classifies voices using centroid-based classification.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check your setup
python check_dependencies.py

# 3. See example workflow
python example_usage.py

# 4. Record training data
python src/sample_recorder.py

# 5. Test voice recognition  
python src/main.py run
```

## 🎯 Overview

This project implements a complete voice recognition pipeline consisting of two main components:

1. **Training Data Collection** (`sample_recorder.py`) - Records voice samples from multiple speakers
2. **Voice Classification** (`main.py`) - Classifies new voice samples using the trained model

The system uses FFT-based frequency band analysis to extract voice characteristics and employs a centroid-based classifier for speaker identification.

## 🔧 Features

- **Real-time audio recording** from microphone input
- **Frequency domain analysis** with custom band edge configuration
- **Automatic voice classification** using machine learning
- **Visual analysis** with time-domain and frequency-domain plots
- **Flexible configuration** for different numbers of speakers and samples
- **CSV output** for predictions and analysis
- **Support for both live and file-based testing**

## 📋 Requirements

### System Dependencies
- **Python 3.7+**
- **PortAudio** (for microphone access)
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
  - macOS: `brew install portaudio`
  - Windows: Usually included with Python audio packages

### Python Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy matplotlib sounddevice python-dotenv
```

### Verify Installation
Run the dependency checker to verify everything is set up correctly:
```bash
python check_dependencies.py
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Daga1404/signals_challenge.git
   cd signals_challenge
   ```

2. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev python3-dev
   
   # macOS
   brew install portaudio
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python check_dependencies.py
   ```

5. **See example workflow:**
   ```bash
   python example_usage.py
   ```

## 📖 Usage

### Step 1: Record Training Data

Use `sample_recorder.py` to collect voice samples from speakers:

```bash
cd src
python sample_recorder.py
```

**What it does:**
- Records 3-second voice samples at 16kHz
- Collects 12 samples per person by default
- Supports 4 speakers: "David", "Gabo", "Dante", "Gal"
- Generates time-domain and frequency-domain visualizations
- Creates a timestamped output directory (e.g., `run_20231215_143022`)

**During recording:**
- Press Enter when prompted to start recording
- Speak clearly for 3 seconds when recording starts
- The system will automatically save WAV files and generate plots

**Output files:**
- `train_p{person}_t{take}.wav` - Training audio files
- `persona_{person}_tomas_time.png` - Time-domain plots
- `{name}_fft.png` - Frequency-domain plots
- `meta.npz` - Metadata for the training session

### Step 2: Test Voice Recognition

Use `main.py` to classify voices using the trained model:

```bash
python main.py <run_directory>
```

**Example:**
```bash
python main.py run_20231215_143022
```

**What it does:**
- Loads the trained model from the specified directory
- Can classify existing test files or record new ones
- Generates predictions with confidence scores
- Creates frequency analysis plots for test samples
- Saves results to `predictions.csv`

**Testing modes:**

1. **File-based testing:** If `test_*.wav` files exist in the directory, classifies them
2. **Live microphone testing:** If no test files exist, records 6 new test samples

## ⚙️ Configuration

### Key Parameters (modifiable in source code):

**Audio Settings:**
```python
EXPECTED_SR = 16000      # Sample rate (Hz)
TAKE_SECONDS = 3.0       # Recording duration per sample
WARMUP_SECONDS = 0.25    # Microphone warmup time
```

**Training Settings:**
```python
N_PERSONS = 4            # Number of speakers
N_TAKES_TRAIN = 12       # Samples per speaker
PERSON_NAMES = ["David", "Gabo", "Dante", "Gal"]  # Speaker names
```

**Feature Extraction:**
```python
FMIN = 50.0              # Minimum frequency (Hz)
FSPLIT = 300.0           # Frequency split point (Hz)
FMAX_BANDS = 3500.0      # Maximum frequency for bands (Hz)
N_LOW = 18               # Low-frequency bands
N_HIGH = 14              # High-frequency bands
```

## 🔬 Technical Details

### Feature Extraction

The system uses a sophisticated frequency band analysis approach:

1. **Windowing:** Applies Hann window to audio samples
2. **FFT Analysis:** Computes frequency spectrum using FFT
3. **Band Division:** 
   - Linear spacing from 50-300 Hz (18 bands) for low frequencies
   - Logarithmic spacing from 300-3500 Hz (14 bands) for high frequencies
4. **Power Calculation:** Computes logarithmic power in each band
5. **Normalization:** RMS normalization per sample

### Classification Algorithm

1. **Feature Standardization:** Z-score normalization across training data
2. **Centroid Calculation:** Computes mean feature vector per speaker
3. **Classification:** Uses Euclidean distance to nearest centroid
4. **Output:** Returns predicted speaker and distance scores

### Audio Processing Pipeline

```
Raw Audio → Windowing → FFT → Band Power → Log Transform → Normalization → Classification
```

## 📁 File Structure

```
signals_challenge/
├── src/
│   ├── sample_recorder.py    # Training data collection
│   └── main.py              # Voice classification
├── requirements.txt         # Python dependencies
├── check_dependencies.py    # Dependency verification script
├── example_usage.py         # Usage examples and workflow guide
├── run_YYYYMMDD_HHMMSS/     # Training session output
│   ├── train_p1_t1.wav      # Training audio files
│   ├── train_p1_t2.wav
│   ├── ...
│   ├── test_1.wav           # Test audio files (optional)
│   ├── persona_1_tomas_time.png  # Time-domain plots
│   ├── david_fft.png        # Frequency-domain plots
│   ├── meta.npz             # Session metadata
│   ├── predictions.csv      # Classification results
│   └── test_1_fft.png       # Test sample analysis
└── README.md                # This file
```

## 📊 Output Analysis

### Training Outputs

- **Time-domain plots:** Show waveform patterns for each speaker
- **Frequency-domain plots:** Display spectral characteristics per speaker
- **WAV files:** Raw audio data for training and testing

### Classification Results

- **Console output:** Real-time predictions with confidence scores
- **CSV file:** Structured results for further analysis
- **Test plots:** Frequency analysis of classified samples

Example prediction output:
```
[pred] test_1.wav -> David  (distancias: 0.234, 0.891, 1.456, 2.123)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Check dependencies first:**
   ```bash
   python check_dependencies.py
   ```
   This will identify any missing packages or system libraries.

2. **PortAudio library not found:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev
   
   # macOS
   brew install portaudio
   ```

3. **Microphone permission denied:**
   - Ensure microphone permissions are granted to terminal/Python
   - Check system audio settings

4. **No audio input detected:**
   - Verify microphone is connected and working
   - Check `sounddevice` can detect your audio device:
     ```python
     import sounddevice as sd
     print(sd.query_devices())
     ```

5. **Poor classification accuracy:**
   - Ensure clear speech during recording
   - Record in a quiet environment
   - Use consistent microphone distance
   - Check that speakers have distinct vocal characteristics

### Performance Tips

- **Record in quiet environments** to minimize background noise
- **Maintain consistent microphone distance** during recording
- **Speak clearly and naturally** for best results
- **Use good quality microphone** for better signal-to-noise ratio

## 🔬 Advanced Usage

### Utility Scripts

The project includes helpful utility scripts:

- **`check_dependencies.py`** - Verifies all required packages are installed
- **`example_usage.py`** - Shows typical workflow and expected outputs
- **`requirements.txt`** - Lists all Python dependencies

### Custom Speaker Configuration

To modify the number of speakers or names:

1. Edit `PERSON_NAMES` in `sample_recorder.py`
2. Update `N_PERSONS` to match the list length
3. Re-record training data with new configuration

### Feature Analysis

The system provides detailed frequency analysis that can be used for:
- Voice characteristic research
- Speaker identification studies
- Audio quality assessment
- Spectral analysis of different voices

### Extending the System

The modular design allows for easy extensions:
- **Additional features:** Modify `features_fft_bands()` function
- **Different classifiers:** Replace centroid-based classification
- **Real-time processing:** Adapt for continuous audio streams
- **Multiple languages:** Test with different language samples

## 📈 Performance Characteristics

- **Sampling Rate:** 16 kHz (suitable for voice)
- **Sample Duration:** 3 seconds per recording
- **Feature Dimensions:** 32 frequency bands (18 low + 14 high)
- **Classification Time:** < 1 second per sample
- **Memory Usage:** Minimal (< 100MB for typical sessions)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature extraction methods
- Alternative classification algorithms
- Real-time processing capabilities
- Enhanced visualization tools
- Support for more audio formats

## 📄 License

This project is open source.

## 🙋 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are properly installed
3. Ensure audio hardware is working correctly
4. Open an issue on the GitHub repository with detailed error information
