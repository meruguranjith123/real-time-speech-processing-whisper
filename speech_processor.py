"""
Speech Processing Module with Whisper Integration
Handles real-time speech recognition, stuttering detection, and next-sentence prediction
"""

import numpy as np
import whisper
import torch
import torchaudio
from typing import List, Tuple, Optional
import re
from collections import deque
import logging
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechProcessor:
    """Main class for processing speech with stuttering detection and prediction"""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the SpeechProcessor
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, or None for auto)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model: {model_size} on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)
        
        # Stuttering detection parameters
        self.repetition_threshold = 2  # Number of repetitions to consider as stutter
        self.recent_words = deque(maxlen=10)  # Track recent words for stutter detection
        
        # Next sentence prediction (simple n-gram approach)
        self.context_words = deque(maxlen=5)  # Keep last 5 words for context
        
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio (default 16000 for Whisper)
            
        Returns:
            Transcribed text
        """
        try:
            # Ensure audio is in the right format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=0)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_tensor = torch.from_numpy(audio_data).float()
                audio_data = resampler(audio_tensor).numpy()
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False
            )
            
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    def detect_stuttering(self, text: str) -> Tuple[str, List[str]]:
        """
        Detect and clean stuttering in transcribed text
        
        Args:
            text: Input text with potential stuttering
            
        Returns:
            Tuple of (cleaned_text, detected_stutters)
        """
        if not text:
            return "", []
        
        # First, detect and remove sentence-level repetitions
        # Split by sentence boundaries (periods, exclamation, question marks)
        sentences = re.split(r'[.!?]+\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        detected_stutters = []
        unique_sentences = []
        sentence_counts = {}
        
        # Track sentence repetitions
        for sentence in sentences:
            # Normalize sentence for comparison (lowercase, remove extra spaces)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            if normalized in sentence_counts:
                sentence_counts[normalized] += 1
            else:
                sentence_counts[normalized] = 1
                unique_sentences.append(sentence)
        
        # Report repeated sentences
        for sent, count in sentence_counts.items():
            if count >= 2:
                detected_stutters.append(f"Sentence repeated {count} times: '{sent[:50]}...'")
        
        # Use unique sentences only
        text = '. '.join(unique_sentences)
        if unique_sentences:
            text += '.' if not text.endswith('.') else ''
        
        # Now handle word-level repetitions within remaining text
        # First, preserve original text for stutter pattern detection
        original_words = re.findall(r'\b\w+\b', text.lower())
        words = original_words.copy()
        cleaned_words = []
        stutter_patterns = []  # Store patterns like "the - 2"
        
        i = 0
        while i < len(words):
            word = words[i]
            count = 1
            
            # Count consecutive repetitions
            j = i + 1
            while j < len(words) and words[j] == word:
                count += 1
                j += 1
            
            # If repetition exceeds threshold, it's a stutter
            if count >= self.repetition_threshold:
                # Format: "word - count" (e.g., "the - 2")
                stutter_patterns.append(f"{word} - {count}")
                detected_stutters.append(f"{word} - {count}")
                cleaned_words.append(word)  # Keep only one instance
                i = j
            else:
                cleaned_words.append(word)
                i += 1
        
        # Also check for partial word repetitions (e.g., "th-th-the")
        text_cleaned = re.sub(r'(\w+)-\1+', r'\1', ' '.join(cleaned_words))
        
        # Remove filler words that might indicate stuttering
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        words_final = [w for w in text_cleaned.split() if w not in filler_words]
        
        # Reconstruct text with proper capitalization
        cleaned_text = ' '.join(words_final)
        
        # Capitalize first letter of sentences
        cleaned_text = re.sub(r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), cleaned_text)
        
        return cleaned_text, detected_stutters
    
    def predict_next_sentences(self, current_text: str, num_predictions: int = 3) -> List[str]:
        """
        Predict possible next sentences based on current context
        
        Args:
            current_text: Current spoken text
            num_predictions: Number of predictions to generate
            
        Returns:
            List of possible next sentence completions
        """
        if not current_text:
            return ["Start speaking..."]
        
        words = current_text.lower().split()
        if not words:
            return ["Start speaking..."]
        
        # Update context
        self.context_words.extend(words[-3:])  # Keep last 3 words
        
        # Simple prediction patterns based on common sentence continuations
        last_word = words[-1] if words else ""
        
        # Common sentence patterns
        predictions = []
        
        # If sentence seems incomplete, suggest completions
        if not current_text.rstrip().endswith(('.', '!', '?')):
            # Generate predictions based on last word
            if last_word in ['i', 'i\'m', 'i\'ve']:
                predictions = [
                    f"{current_text} am going to explain this concept.",
                    f"{current_text} want to discuss this topic.",
                    f"{current_text} think this is important."
                ]
            elif last_word in ['we', 'we\'re', 'we\'ve']:
                predictions = [
                    f"{current_text} are going to explore this.",
                    f"{current_text} need to understand this better.",
                    f"{current_text} can see the implications."
                ]
            elif last_word in ['this', 'that', 'these', 'those']:
                predictions = [
                    f"{current_text} is a key concept.",
                    f"{current_text} demonstrates the principle.",
                    f"{current_text} shows us how it works."
                ]
            elif last_word in ['the', 'a', 'an']:
                predictions = [
                    f"{current_text} main idea is clear.",
                    f"{current_text} concept is important.",
                    f"{current_text} solution works well."
                ]
            else:
                # Generic completions
                predictions = [
                    f"{current_text} is important to understand.",
                    f"{current_text} helps us see the bigger picture.",
                    f"{current_text} demonstrates the key principle."
                ]
        else:
            # Sentence is complete, suggest new sentence starters
            predictions = [
                "Let me explain this further.",
                "This leads us to the next point.",
                "We should also consider this aspect."
            ]
        
        return predictions[:num_predictions]
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Complete processing pipeline: transcribe, clean stuttering, predict next sentences
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with transcription, cleaned text, stutters, and predictions
        """
        # Transcribe
        raw_text = self.transcribe_audio(audio_data, sample_rate)
        
        if not raw_text:
            return {
                "raw_text": "",
                "cleaned_text": "",
                "stutters": [],
                "predictions": ["Start speaking..."]
            }
        
        # Detect and clean stuttering
        cleaned_text, stutters = self.detect_stuttering(raw_text)
        
        # Predict next sentences based on RAW text (with stutters) for better context
        # This gives predictions that understand the full context including stutters
        predictions = self.predict_next_sentences(raw_text)
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "stutters": stutters,
            "predictions": predictions
        }
    
    def reset_context(self):
        """Reset the context for new conversation"""
        self.recent_words.clear()
        self.context_words.clear()

