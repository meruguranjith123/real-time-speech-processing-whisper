# Real-Time Speech Processing with Whisper

A Python module and web demo for real-time speech recognition with stuttering detection and next-sentence prediction using OpenAI's Whisper model.

## Features

- 🎤 **Real-time Speech Recognition**: Uses Whisper AI for accurate transcription
- 🧹 **Stuttering Detection & Cleaning**: Automatically detects and removes stutters and repetitions
- 🔮 **Next Sentence Prediction**: Predicts possible next sentences based on current speech context
- 🌐 **Web Interface**: Beautiful, modern web UI for real-time audio processing
- 📊 **Audio Visualization**: Real-time audio waveform visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ffmpeg (required for audio format conversion)
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd SBU-CSE-570
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Note**: The first time you run the application, Whisper will download the model (base model by default). This may take a few minutes.

## Usage

### Running the Web Demo

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Click "Start Recording"** to begin capturing audio from your microphone
4. **Speak naturally** - the system will:
   - Transcribe your speech in real-time
   - Clean stutters and repetitions
   - Show predicted next sentences

5. **Click "Stop Recording"** when done

### Using the Python Module

You can also use the `SpeechProcessor` class directly in your Python code:

```python
from speech_processor import SpeechProcessor
import numpy as np

# Initialize processor
processor = SpeechProcessor(model_size="base")

# Process audio (numpy array, 16kHz sample rate)
audio_data = np.array([...])  # Your audio data
result = processor.process_audio(audio_data, sample_rate=16000)

print(f"Raw text: {result['raw_text']}")
print(f"Cleaned text: {result['cleaned_text']}")
print(f"Detected stutters: {result['stutters']}")
print(f"Predictions: {result['predictions']}")
```

### Standalone Demo Script

You can also test the module with a standalone command-line demo:

```bash
python demo_standalone.py
```

This will:
1. Record 5 seconds of audio from your microphone
2. Process it and display all results
3. Show transcription, cleaned text, stutters, and predictions

## Project Structure

```
SBU-CSE-570/
├── speech_processor.py    # Main speech processing module
├── app.py                 # Flask web server
├── demo_standalone.py     # Standalone command-line demo
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

### Speech Recognition
- Uses OpenAI's Whisper model for transcription
- Supports multiple model sizes: tiny, base, small, medium, large
- Default: "base" model (good balance of speed and accuracy)

### Stuttering Detection
- Detects consecutive word repetitions (e.g., "the the the")
- Removes partial word repetitions (e.g., "th-th-the")
- Filters out filler words (um, uh, er, etc.)
- Returns cleaned text with detected stutters listed

### Next Sentence Prediction
- Uses context from recent words
- Generates likely sentence completions
- Adapts predictions based on sentence structure
- Provides 3 predictions by default

## Configuration

### Model Size
You can change the Whisper model size in `app.py`:

```python
speech_processor = SpeechProcessor(model_size="base")
```

Available options:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

### Stuttering Threshold
Adjust the repetition threshold in `speech_processor.py`:

```python
self.repetition_threshold = 2  # Number of repetitions to consider as stutter
```

## Browser Compatibility

The web interface uses:
- Web Audio API for microphone access
- MediaRecorder API for audio capture
- WebSocket (Socket.IO) for real-time communication

Works best in:
- Chrome/Edge (recommended)
- Firefox
- Safari (may have limitations)

## Troubleshooting

### Microphone Access Issues
- Ensure your browser has microphone permissions
- Check browser settings for microphone access
- Try refreshing the page

### Model Download Issues
- First run will download the model (~500MB for base model)
- Ensure stable internet connection
- Model is cached after first download

### Audio Processing Errors
- Check that your microphone is working
- Ensure sample rate is 16kHz (handled automatically)
- Check server logs for detailed error messages

## Performance Notes

- **Model Size**: Larger models are more accurate but slower
- **Real-time Processing**: Processing happens in ~1 second chunks
- **GPU Support**: Automatically uses GPU if available (CUDA)
- **CPU Usage**: Base model runs well on CPU

## License

This project is for educational purposes.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Flask and Socket.IO for web framework
- All open-source contributors

