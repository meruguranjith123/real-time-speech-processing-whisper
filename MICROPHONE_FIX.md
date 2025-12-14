# 🎤 Microphone Permission Fix Guide

## Quick Fix Steps

### 1. **Chrome/Edge (Recommended)**
1. Look for the **lock icon (🔒)** or **camera/microphone icon** in the address bar
2. Click it
3. Find "Microphone" in the dropdown
4. Change from "Block" to **"Allow"**
5. **Refresh the page** (F5 or Cmd+R)
6. Click "Start Recording" again

### 2. **Firefox**
1. Click the **shield icon** in the address bar
2. Go to **Permissions** tab
3. Find **Microphone**
4. Click **Allow**
5. Refresh the page

### 3. **Safari**
1. Safari menu → **Settings** → **Websites**
2. Click **Microphone** in the left sidebar
3. Find `localhost:5001` in the list
4. Change to **Allow**
5. Refresh the page

### 4. **System Settings (macOS)**
1. Open **System Preferences** → **Security & Privacy**
2. Click **Privacy** tab
3. Select **Microphone** from the left
4. Make sure your browser (Chrome/Firefox) is **checked**
5. If not, check the box next to your browser
6. Restart your browser

## Common Issues

### Issue: "Permission Denied"
**Solution:**
- Check browser address bar for lock/microphone icon
- Allow microphone access
- Refresh page
- Try again

### Issue: "No Microphone Found"
**Solution:**
- Check if microphone is connected
- Check System Preferences → Sound → Input
- Make sure microphone is selected and working
- Test in another app (like Voice Memos)

### Issue: "Microphone in Use"
**Solution:**
- Close other apps using microphone (Zoom, Teams, etc.)
- Check Activity Monitor for apps using audio
- Restart browser
- Try again

### Issue: Browser Doesn't Support Microphone
**Solution:**
- Use **Chrome** or **Edge** (best support)
- Firefox also works well
- Safari has limited support

## Testing Your Microphone

### Test in Browser:
1. Go to: https://www.onlinemictest.com/
2. Click "Test Mic"
3. If it works there, the issue is with permissions, not hardware

### Test in System:
1. Open **Voice Memos** (macOS)
2. Try recording
3. If it works, microphone is fine - it's a browser permission issue

## Step-by-Step Fix

1. **Close all browser tabs** for localhost:5001
2. **Open browser settings**:
   - Chrome: `chrome://settings/content/microphone`
   - Edge: `edge://settings/content/microphone`
3. **Remove localhost from blocked sites**
4. **Open new tab** → Go to `http://localhost:5001`
5. **Click "Start Recording"**
6. **When prompted**, click **"Allow"**
7. **If no prompt**, check address bar icon

## Still Not Working?

### Check Browser Console:
1. Press **F12** (or Cmd+Option+I on Mac)
2. Go to **Console** tab
3. Look for red error messages
4. Share the error message for help

### Check Server:
```bash
# Make sure server is running
curl http://localhost:5001/health

# If not running, start it:
./start_server.sh
```

### Try Different Browser:
- If Chrome doesn't work, try Edge
- If Edge doesn't work, try Firefox
- Safari has limited support

## Verification

After fixing permissions, you should see:
- ✅ Browser asks for permission (first time)
- ✅ Address bar shows microphone icon (not blocked)
- ✅ Status changes to "🔴 Recording..." when you click Start
- ✅ Audio visualization bars start moving
- ✅ No error alerts

## Need More Help?

1. Check browser console (F12) for specific errors
2. Verify server is running: `curl http://localhost:5001/health`
3. Try a different browser
4. Check system microphone settings
5. Restart browser completely

---

**Most common fix:** Click the lock/microphone icon in address bar → Allow → Refresh page





