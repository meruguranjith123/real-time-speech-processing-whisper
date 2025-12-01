"""
Flask web application for real-time speech processing demo
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import io
import wave
import tempfile
import os
from speech_processor import SpeechProcessor
import logging
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    # Test if ffmpeg backend works
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
        if result.returncode != 0:
            logger.warning("ffmpeg found but may not be working correctly")
    except:
        logging.warning("ffmpeg may not be properly configured")
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, audio conversion may be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize speech processor (lazy loading - will load on first use)
speech_processor = None

def get_speech_processor():
    """Lazy load the speech processor"""
    global speech_processor
    if speech_processor is None:
        logger.info("Loading Whisper model (this may take a moment)...")
        try:
            speech_processor = SpeechProcessor(model_size="base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    return speech_processor

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@socketio.on('audio_data')
def handle_audio(audio_data):
    """
    Handle incoming audio data from client
    
    Args:
        audio_data: Base64 encoded audio data (PCM or WebM format from browser)
    """
    try:
        processor = get_speech_processor()
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data['data'])
        
        # Check if audio is in PCM format (from Web Audio API)
        audio_format = audio_data.get('format', 'webm')
        
        if audio_format == 'pcm':
            # Direct PCM processing - no ffmpeg needed!
            try:
                # Convert base64 to int16 array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                sample_rate = audio_data.get('sample_rate', 16000)
                
                # Resample to 16kHz if needed (Whisper requirement)
                if sample_rate != 16000:
                    try:
                        import scipy.signal
                        num_samples = int(len(audio_array) * 16000 / sample_rate)
                        audio_array = scipy.signal.resample(audio_array, num_samples)
                        sample_rate = 16000
                    except ImportError:
                        logger.warning("scipy not available, skipping resampling")
                
                logger.info(f"Processed PCM audio: {len(audio_array)} samples at {sample_rate}Hz")
            except Exception as pcm_error:
                logger.error(f"PCM processing error: {pcm_error}")
                emit('error', {
                    'message': f'PCM audio processing failed: {str(pcm_error)}. Please try again.'
                })
                return
        elif audio_format == 'webm' and PYDUB_AVAILABLE:
            # Handle WebM/Opus audio from browser (fallback)
            temp_webm_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
                    temp_webm.write(audio_bytes)
                    temp_webm_path = temp_webm.name
                
                # Try multiple methods to decode audio
                methods = [
                    ("webm", "webm format"),
                    (None, "auto-detect format"),
                    ("opus", "opus format"),
                ]
                
                audio_segment = None
                last_error = None
                for format_spec, method_name in methods:
                    try:
                        logger.info(f"Trying to load audio as {method_name}...")
                        if format_spec:
                            audio_segment = AudioSegment.from_file(temp_webm_path, format=format_spec)
                        else:
                            audio_segment = AudioSegment.from_file(temp_webm_path)
                        logger.info(f"Successfully loaded audio using {method_name}")
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed to load as {method_name}: {str(e)[:100]}")
                        continue
                
                if audio_segment is None:
                    error_details = str(last_error) if last_error else "Unknown error"
                    raise Exception(f"Could not decode audio. Last error: {error_details[:200]}")
                
                # Convert to mono and 16kHz (Whisper requirement)
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(16000)
                
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1]
                if audio_segment.sample_width == 2:
                    audio_array = audio_array / 32768.0
                elif audio_segment.sample_width == 4:
                    audio_array = audio_array / 2147483648.0
                else:
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                
                if len(audio_array) == 0:
                    raise Exception("Audio array is empty after processing")
                
                sample_rate = 16000
                
            except Exception as e:
                logger.error(f"Error processing audio with pydub: {e}")
                emit('error', {
                    'message': f'Audio processing failed: {str(e)[:200]}. Please refresh and try again.'
                })
                return
            finally:
                if temp_webm_path and os.path.exists(temp_webm_path):
                    try:
                        os.unlink(temp_webm_path)
                    except:
                        pass
        else:
            logger.error(f"Unknown or unsupported audio format: {audio_format}")
            emit('error', {'message': f'Unsupported audio format: {audio_format}'})
            return
        
        # Process audio
        result = processor.process_audio(
            audio_array,
            sample_rate=sample_rate
        )
        
        # Emit results back to client
        emit('transcription_result', result)
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit('error', {'message': str(e)})

@socketio.on('reset')
def handle_reset():
    """Reset the speech processor context"""
    try:
        processor = get_speech_processor()
        processor.reset_context()
        emit('reset_complete', {'status': 'ok'})
    except Exception as e:
        logger.error(f"Error resetting context: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting Flask server on port {port}...")
    try:
        socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use. Trying port {port + 1}...")
            socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port + 1, allow_unsafe_werkzeug=True)
        else:
            raise

