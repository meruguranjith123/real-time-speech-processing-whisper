"""
Standalone demo script for testing the SpeechProcessor module
This script demonstrates how to use the module with a microphone input
"""

import pyaudio
import numpy as np
from speech_processor import SpeechProcessor
import time

def record_audio(duration=5, sample_rate=16000, chunk_size=1024):
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate
        chunk_size: Chunk size for recording
        
    Returns:
        numpy array of audio data
    """
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    
    print(f"Recording for {duration} seconds...")
    frames = []
    
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    return audio_data

def main():
    """Main demo function"""
    print("=" * 60)
    print("Speech Processing Demo - Standalone Version")
    print("=" * 60)
    print("\nThis demo will:")
    print("1. Record 5 seconds of audio from your microphone")
    print("2. Transcribe it using Whisper")
    print("3. Detect and clean stutters")
    print("4. Predict next sentences")
    print("\nPress Enter to start recording...")
    input()
    
    # Initialize processor
    print("\nInitializing Whisper model (this may take a moment)...")
    processor = SpeechProcessor(model_size="base")
    
    # Record audio
    audio_data = record_audio(duration=5)
    
    print("\nProcessing audio...")
    result = processor.process_audio(audio_data, sample_rate=16000)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📝 Raw Transcription:")
    print(f"   {result['raw_text']}")
    
    print(f"\n✨ Cleaned Text (Stutters Removed):")
    print(f"   {result['cleaned_text']}")
    
    if result['stutters']:
        print(f"\n⚠️  Detected Stutters:")
        for stutter in result['stutters']:
            print(f"   - {stutter}")
    else:
        print(f"\n✅ No stutters detected")
    
    print(f"\n🔮 Predicted Next Sentences:")
    for i, prediction in enumerate(result['predictions'], 1):
        print(f"   {i}. {prediction}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  - Microphone connected and working")
        print("  - pyaudio installed: pip install pyaudio")
        print("  - All dependencies from requirements.txt installed")

