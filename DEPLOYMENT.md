# Deployment Guide

This Flask application can be deployed to various cloud platforms. GitHub Pages only hosts static sites, so you'll need a platform that supports Python/Flask applications.

## 🚀 Recommended Platforms

### 1. **Render** (Easiest - Free Tier Available)
✅ **Best for beginners**

1. Push your code to GitHub
2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Settings:
   - **Name**: speech-processor (or any name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free (or paid for better performance)

6. Add environment variable (optional):
   - `PYTHON_VERSION=3.9.18`

**Note**: Render free tier spins down after inactivity, so first request may be slow.

---

### 2. **Railway** (Good Free Tier)
✅ **Fast deployment, good free tier**

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and installs dependencies
6. Add start command in settings: `python app.py`

**Note**: Free tier includes $5/month credit.

---

### 3. **Heroku** (Classic, but requires credit card)
⚠️ **Requires credit card for free tier**

1. Install Heroku CLI: `brew install heroku/brew/heroku`
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. The Procfile is already configured!

**Note**: Heroku free tier was discontinued, now requires paid plan.

---

### 4. **Fly.io** (Good for global distribution)
✅ **Free tier available**

1. Install Fly CLI: `brew install flyctl`
2. Login: `fly auth login`
3. Launch: `fly launch`
4. Follow prompts

---

### 5. **PythonAnywhere** (Simple Python hosting)
✅ **Free tier available**

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files via web interface or Git
3. Configure web app in dashboard
4. Set WSGI file (see below)

---

### 6. **Replit** (Great for demos)
✅ **Free tier, instant deployment**

1. Go to [replit.com](https://replit.com)
2. Click "Create Repl" → "Import from GitHub"
3. Select your repository
4. Click "Run"

---

## 📝 Important Notes for Deployment

### Port Configuration
Most platforms set the port via environment variable. Update `app.py`:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Flask server...")
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
```

### Environment Variables
Some platforms may need:
- `PORT`: Automatically set by platform
- `PYTHON_VERSION`: Python version (3.9+)

### Model Size
For faster deployment, consider using `tiny` model:
```python
speech_processor = SpeechProcessor(model_size="tiny")
```

### Buildpacks (if needed)
Some platforms may need:
- Python buildpack
- ffmpeg buildpack (for audio conversion)

---

## 🔧 Platform-Specific Configurations

### For Render
Create `render.yaml`:
```yaml
services:
  - type: web
    name: speech-processor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.18
```

### For Railway
Railway auto-detects, but you can add `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### For PythonAnywhere
Create `wsgi.py`:
```python
import sys
path = '/home/yourusername/speech-processor'
if path not in sys.path:
    sys.path.append(path)

from app import app, socketio

application = app

if __name__ == "__main__":
    socketio.run(app)
```

---

## 🐳 Docker Deployment (Advanced)

If you want to use Docker, create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Then deploy to:
- Docker Hub + Any container platform
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances

---

## ✅ Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] `Procfile` created (for Heroku/Railway)
- [ ] Port configuration updated (if needed)
- [ ] Environment variables set (if needed)
- [ ] Test locally before deploying

---

## 🆘 Troubleshooting Deployment

### Issue: "Module not found"
- Ensure `requirements.txt` has all dependencies
- Check build logs for installation errors

### Issue: "Port already in use"
- Use environment variable for port: `os.environ.get('PORT', 5000)`

### Issue: "Whisper model download fails"
- Model downloads on first use
- Ensure platform has internet access
- Consider pre-downloading model in Docker image

### Issue: "ffmpeg not found"
- Add ffmpeg to buildpack/system packages
- Or use platform that includes it (Render, Railway)

---

## 📊 Platform Comparison

| Platform | Free Tier | Ease | Speed | Best For |
|----------|-----------|------|-------|----------|
| Render | ✅ Yes | ⭐⭐⭐⭐⭐ | Medium | Beginners |
| Railway | ✅ Yes | ⭐⭐⭐⭐ | Fast | Quick deploys |
| Fly.io | ✅ Yes | ⭐⭐⭐ | Fast | Global apps |
| Heroku | ❌ Paid | ⭐⭐⭐⭐ | Fast | Production |
| Replit | ✅ Yes | ⭐⭐⭐⭐⭐ | Fast | Demos |
| PythonAnywhere | ✅ Yes | ⭐⭐⭐ | Medium | Simple apps |

---

## 🎯 Recommended: Render or Railway

For this project, I recommend **Render** or **Railway**:
- Easy setup
- Free tier available
- Good documentation
- Automatic HTTPS
- Easy GitHub integration





