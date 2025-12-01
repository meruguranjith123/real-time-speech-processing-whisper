# Troubleshooting Guide

## Issue: Localhost showing empty page

### Step 1: Check if server is running

```bash
# Check if Python process is running
ps aux | grep "python.*app.py"

# Or check if port 5000 is in use
lsof -i :5000
```

### Step 2: Start the server

```bash
cd /Users/meruguranjith/Downloads/SBU-CSE-570
python3 app.py
```

You should see:
```
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

### Step 3: Test basic connectivity

Open in browser:
- `http://localhost:5000/health` - Should return `{"status": "ok"}`
- `http://localhost:5000/` - Should show the web interface

### Step 4: Check browser console

1. Open browser Developer Tools (F12 or Cmd+Option+I)
2. Go to Console tab
3. Look for any JavaScript errors
4. Check Network tab to see if files are loading

### Step 5: Check server logs

Look at the terminal where you ran `python3 app.py` for any error messages.

## Common Issues

### Issue: "Module not found"
**Solution:**
```bash
pip3 install -r requirements.txt
```

### Issue: "Address already in use"
**Solution:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or change port in app.py (last line):
socketio.run(app, debug=True, host='0.0.0.0', port=5001)
```

### Issue: "Template not found"
**Solution:**
- Make sure `templates/index.html` exists
- Check you're running from the project root directory

### Issue: Page loads but is blank/white
**Possible causes:**
1. JavaScript error - Check browser console
2. Socket.IO not connecting - Check Network tab for WebSocket connection
3. CSS not loading - Check if styles are applied

### Issue: "Whisper model loading..."
**This is normal on first use:**
- Model downloads automatically (~500MB)
- Takes a few minutes on first run
- Subsequent runs are faster

## Quick Test

Run the test script:
```bash
python3 test_server.py
```

This will verify all dependencies are installed correctly.

## Still having issues?

1. Check Python version:
   ```bash
   python3 --version  # Should be 3.8+
   ```

2. Verify all files exist:
   ```bash
   ls -la templates/index.html
   ls -la app.py
   ls -la speech_processor.py
   ```

3. Try a minimal test:
   ```bash
   python3 -c "from flask import Flask; app = Flask(__name__); print('OK')"
   ```

