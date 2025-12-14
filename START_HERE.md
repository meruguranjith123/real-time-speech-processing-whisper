# 🚀 START HERE - Quick Demo Guide

## ✅ Server is Running!

Your website is **LIVE** and ready for demo!

### 🌐 Access Your Website

**Open in your browser:**
```
http://localhost:5001
```

### 🎤 How to Use

1. **Open the website** in your browser (Chrome/Edge recommended)
2. **Click "Start Recording"** button
3. **Allow microphone access** when prompted
4. **Start speaking** - the system will:
   - Transcribe your speech in real-time
   - Clean stutters automatically
   - Show predicted next sentences

### 📊 What You'll See

- **Raw Transcription**: What Whisper heard
- **Cleaned Text**: Stutters removed
- **Detected Stutters**: List of detected repetitions
- **Predicted Next Sentences**: 3 suggestions based on your speech

### 🛠️ Server Management

**Start Server:**
```bash
./start_server.sh
```

**Stop Server:**
```bash
kill $(cat server.pid)
```

**Check Server Status:**
```bash
curl http://localhost:5001/health
```

**View Logs:**
```bash
tail -f server.log
```

### ⚠️ Important Notes

- **First Recording**: Whisper model loads on first use (~30 seconds)
- **Port**: Using port 5001 (5000 is used by macOS)
- **Browser**: Use Chrome or Edge for best compatibility
- **Microphone**: Make sure to allow microphone access

### 🐛 Troubleshooting

**Website not loading?**
- Check if server is running: `ps aux | grep app.py`
- Check logs: `tail server.log`
- Restart: `./start_server.sh`

**Microphone not working?**
- Check browser permissions
- Try a different browser
- Check system microphone settings

**No transcription?**
- Wait for Whisper model to load (first time only)
- Speak clearly and wait a few seconds
- Check browser console (F12) for errors

### 📝 Demo Tips

1. **Test with stuttering**: Try saying "the the the main idea is..."
2. **Use filler words**: Say "um uh the concept is..."
3. **Natural speech**: Speak normally and see predictions
4. **Show cleaning**: Compare raw vs cleaned text

---

**🎉 Your demo is ready! Open http://localhost:5001 and start recording!**





