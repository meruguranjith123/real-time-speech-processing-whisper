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
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, audio conversion may be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize speech processor
speech_processor = SpeechProcessor(model_size="base")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@socketio.on('audio_data')
def handle_audio(audio_data):
    """
    Handle incoming audio data from client
    
    Args:
        audio_data: Base64 encoded audio data (WebM format from browser)
    """
    try:
        logger.info(f"Received audio data: {len(audio_data.get('data', ''))} characters in base64")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data['data'])
        logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
        
        if len(audio_bytes) < 100:
            logger.warning(f"Audio chunk too small: {len(audio_bytes)} bytes, skipping")
            emit('transcription_result', {
                "raw_text": "",
                "cleaned_text": "",
                "stutters": [],
                "predictions": ["Audio chunk too small, please speak louder or longer"]
            })
            return
        
        # Check audio format
        audio_format = audio_data.get('format', 'webm')
        logger.info(f"Received audio format: {audio_format}, data length: {len(audio_bytes)} bytes")
        
        # Handle PCM audio (preferred - direct from Web Audio API)
        if audio_format == 'pcm':
            try:
                # Convert base64 to int16 array
                # The base64 string needs to be decoded properly
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                sample_rate = audio_data.get('sample_rate', 16000)
                
                logger.info(f"Processed PCM audio: {len(audio_array)} samples at {sample_rate}Hz, max amplitude: {np.max(np.abs(audio_array))}")
                
                # Check if we have enough samples (at least 0.5 seconds at 16kHz)
                if len(audio_array) < 8000:
                    logger.warning(f"Audio chunk too short: {len(audio_array)} samples")
                    emit('transcription_result', {
                        "raw_text": "",
                        "cleaned_text": "",
                        "stutters": [],
                        "predictions": ["Audio chunk too short. Please speak longer."]
                    })
                    return
                
                # Check if audio has actual content (not just silence)
                if np.max(np.abs(audio_array)) < 0.01:
                    logger.warning("Audio appears to be mostly silence")
                    emit('transcription_result', {
                        "raw_text": "",
                        "cleaned_text": "",
                        "stutters": [],
                        "predictions": ["No audio detected. Please speak louder or check your microphone."]
                    })
                    return
                    
            except Exception as pcm_error:
                logger.error(f"PCM processing error: {pcm_error}", exc_info=True)
                emit('error', {'message': f'PCM audio processing failed: {str(pcm_error)}'})
                return
        
        # Handle WebM/Opus audio from browser (fallback)
        elif audio_format == 'webm' and PYDUB_AVAILABLE:
            # Use pydub to convert WebM to WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
                temp_webm.write(audio_bytes)
                temp_webm_path = temp_webm.name
            
            try:
                # Load and convert audio
                logger.info(f"Loading audio from WebM file: {temp_webm_path}")
                audio_segment = AudioSegment.from_file(temp_webm_path, format="webm")
                logger.info(f"Loaded audio: {len(audio_segment)}ms, {audio_segment.frame_rate}Hz, {audio_segment.channels} channels")
                
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
                
                sample_rate = 16000
                logger.info(f"Processed audio array: {len(audio_array)} samples, max amplitude: {np.max(np.abs(audio_array))}")
                
                # Check if audio has actual content (not just silence)
                if np.max(np.abs(audio_array)) < 0.01:
                    logger.warning("Audio appears to be mostly silence")
                    emit('transcription_result', {
                        "raw_text": "",
                        "cleaned_text": "",
                        "stutters": [],
                        "predictions": ["No audio detected. Please speak louder or check your microphone."]
                    })
                    return
                    
            except Exception as e:
                logger.error(f"Error processing WebM audio: {e}")
                emit('error', {'message': f'Error processing WebM audio: {str(e)}. Try using Chrome or Edge browser.'})
                return
            finally:
                # Clean up temp file
                if os.path.exists(temp_webm_path):
                    os.unlink(temp_webm_path)
        else:
            # No valid format or pydub not available
            if audio_format == 'webm' and not PYDUB_AVAILABLE:
                logger.error("Cannot process WebM audio without pydub. Please install: pip install pydub")
                emit('error', {'message': 'Audio processing requires pydub. Install with: pip install pydub'})
            else:
                logger.error(f"Unknown or unsupported audio format: {audio_format}")
                emit('error', {'message': f'Unsupported audio format: {audio_format}. Please use PCM or WebM format.'})
            return
        
        # Process audio
        logger.info(f"Processing audio with Whisper: {len(audio_array)} samples at {sample_rate}Hz")
        result = speech_processor.process_audio(
            audio_array,
            sample_rate=sample_rate
        )
        
        logger.info(f"Transcription result - Raw: '{result.get('raw_text', '')}', Cleaned: '{result.get('cleaned_text', '')}'")
        
        # Emit results back to client
        emit('transcription_result', result)
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit('error', {'message': str(e)})

@socketio.on('reset')
def handle_reset():
    """Reset the speech processor context"""
    speech_processor.reset_context()
    emit('reset_complete', {'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on localhost:{port}...")
    socketio.run(app, debug=True, host='127.0.0.1', port=port, allow_unsafe_werkzeug=True)

