# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install ffmpeg (required for audio conversion)
# macOS:
brew install ffmpeg

# Linux (Ubuntu/Debian):
sudo apt-get install ffmpeg

# Linux (CentOS/RHEL):
sudo yum install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Step 2: Start the Web Server

```bash
python app.py
# or
python3 app.py
```

You should see:
```
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

### Step 3: Open in Browser

1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. Click **"Start Recording"** when prompted for microphone access
4. Start speaking!

## 📋 What to Expect

- **First Run**: Whisper will download the model (~500MB for base model) - this only happens once
- **Real-time Processing**: Audio is processed in ~1 second chunks
- **Results Displayed**:
  - Raw transcription (what Whisper heard)
  - Cleaned text (stutters removed)
  - Detected stutters (if any)
  - Predicted next sentences (3 suggestions)

## 🧪 Test the Module Directly

You can also test without the web interface:

```bash
python demo_standalone.py
```

This will record 5 seconds of audio and show all results.

## ⚠️ Troubleshooting

### "Module not found" errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`

### Microphone not working
- Check browser permissions for microphone access
- Try a different browser (Chrome/Edge recommended)

### Audio processing errors
- Ensure ffmpeg is installed and in your PATH
- Check that pydub is installed: `pip install pydub`

### Model download issues
- First run requires internet connection to download Whisper model
- Model is cached after first download

## 📝 Notes

- The base model is a good balance of speed and accuracy
- For better accuracy (slower), change `model_size="base"` to `"small"` or `"medium"` in `app.py`
- For faster processing (less accurate), use `"tiny"`





