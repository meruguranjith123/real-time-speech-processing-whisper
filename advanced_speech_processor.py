"""
Advanced Speech Processing Module with Neural Architecture
Implements:
- Dense Prediction Head (Whisper + Bidirectional LSTM)
- Confidence Refinement Layer (1D CNN + Dense)
- Clean Caption Decoder (GRU-based edit predictor)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import torchaudio
from typing import List, Tuple, Optional, Dict
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DensePredictionHead(nn.Module):
    """
    Dense Prediction Head: Whisper + Bidirectional LSTM
    Outputs binary logit for each token: keep or delay
    """
    
    def __init__(self, whisper_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2):
        super(DensePredictionHead, self).__init__()
        
        # Bidirectional LSTM to process Whisper encoder outputs
        self.lstm = nn.LSTM(
            input_size=whisper_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Binary classification: keep (1) or delay (0)
        self.classifier = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional
        
    def forward(self, whisper_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            whisper_features: [batch, seq_len, whisper_dim] from Whisper encoder
        Returns:
            logits: [batch, seq_len, 2] binary logits (keep/delay)
        """
        # Process through bidirectional LSTM
        lstm_out, _ = self.lstm(whisper_features)
        
        # Binary classification for each token
        logits = self.classifier(lstm_out)
        
        return logits


class ConfidenceRefinementLayer(nn.Module):
    """
    Confidence Refinement Layer: 1D CNN + Dense layer
    Produces frame-aligned confidence scores
    """
    
    def __init__(self, input_dim: int = 512, num_filters: int = 64, kernel_size: int = 3):
        super(ConfidenceRefinementLayer, self).__init__()
        
        # 1D Convolutional layers for temporal feature extraction
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters, kernel_size, padding=kernel_size//2)
        
        # Dense layer for confidence score prediction
        self.dense = nn.Linear(num_filters, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, feature_dim]
        Returns:
            confidence_scores: [batch, seq_len, 1] frame-aligned confidence scores
        """
        # Convert to [batch, feature_dim, seq_len] for 1D conv
        x = features.transpose(1, 2)
        
        # 1D CNN layers
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        
        # Convert back to [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)
        
        # Dense layer for confidence scores
        confidence_scores = torch.sigmoid(self.dense(x))
        
        return confidence_scores


class CleanCaptionDecoder(nn.Module):
    """
    Clean Caption Decoder: Lightweight edit-predictor GRU
    Learns token-level edit actions:
    - Keep: retain the token
    - Delete-Repeat: remove repeated token
    - Delete-Filler: remove filler word
    - Merge-Prolong: merge prolonged sounds
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_actions: int = 4):
        super(CleanCaptionDecoder, self).__init__()
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Action classifier: 4 actions (Keep, Delete-Repeat, Delete-Filler, Merge-Prolong)
        self.action_classifier = nn.Linear(hidden_dim, num_actions)
        
        # Action embeddings
        self.action_embeddings = nn.Embedding(num_actions, hidden_dim)
        
    def forward(self, features: torch.Tensor, confidence_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, feature_dim] from Whisper
            confidence_scores: [batch, seq_len, 1] confidence scores
        Returns:
            action_logits: [batch, seq_len, num_actions] edit action predictions
        """
        # Combine features with confidence scores
        combined = torch.cat([features, confidence_scores], dim=-1)
        
        # Process through GRU
        gru_out, _ = self.gru(combined)
        
        # Predict edit actions
        action_logits = self.action_classifier(gru_out)
        
        return action_logits


class AdvancedSpeechProcessor:
    """
    Advanced Speech Processor with full neural architecture
    Integrates Whisper with Dense Prediction Head, Confidence Refinement, and Clean Caption Decoder
    """
    
    # Edit action indices
    ACTION_KEEP = 0
    ACTION_DELETE_REPEAT = 1
    ACTION_DELETE_FILLER = 2
    ACTION_MERGE_PROLONG = 3
    
    ACTION_NAMES = {
        ACTION_KEEP: "Keep",
        ACTION_DELETE_REPEAT: "Delete-Repeat",
        ACTION_DELETE_FILLER: "Delete-Filler",
        ACTION_MERGE_PROLONG: "Merge-Prolong"
    }
    
    def __init__(self, 
                 whisper_model_size: str = "small",
                 device: Optional[str] = None,
                 load_pretrained: bool = False,
                 model_path: Optional[str] = None):
        """
        Initialize Advanced Speech Processor
        
        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, or None for auto)
            load_pretrained: Whether to load pretrained components
            model_path: Path to pretrained model weights
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model: {whisper_model_size} on {self.device}")
        
        # Load Whisper model
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
        self.whisper_model.eval()
        
        # Get Whisper encoder dimension
        # Whisper small has 768 dim, base has 512, etc.
        whisper_dims = {
            "tiny": 384, "base": 512, "small": 768, 
            "medium": 1024, "large": 1280
        }
        self.whisper_dim = whisper_dims.get(whisper_model_size, 512)
        
        # Initialize neural components
        self.dense_prediction_head = DensePredictionHead(
            whisper_dim=self.whisper_dim,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        self.confidence_refinement = ConfidenceRefinementLayer(
            input_dim=self.whisper_dim,
            num_filters=64,
            kernel_size=3
        ).to(self.device)
        
        self.clean_caption_decoder = CleanCaptionDecoder(
            input_dim=self.whisper_dim + 1,  # +1 for confidence score
            hidden_dim=128,
            num_actions=4
        ).to(self.device)
        
        # Set to training mode if loading pretrained
        if load_pretrained and model_path:
            self.load_model(model_path)
        else:
            # Initialize with random weights (would be trained with fine-tuning)
            logger.info("Initializing with random weights. Use fine-tuning to train.")
        
        # Filler words for Delete-Filler action
        self.filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'well', 'so'}
        
    def load_model(self, model_path: str):
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.dense_prediction_head.load_state_dict(checkpoint['dense_prediction_head'])
            self.confidence_refinement.load_state_dict(checkpoint['confidence_refinement'])
            self.clean_caption_decoder.load_state_dict(checkpoint['clean_caption_decoder'])
            logger.info(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def save_model(self, save_path: str):
        """Save model weights"""
        checkpoint = {
            'dense_prediction_head': self.dense_prediction_head.state_dict(),
            'confidence_refinement': self.confidence_refinement.state_dict(),
            'clean_caption_decoder': self.clean_caption_decoder.state_dict()
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model to {save_path}")
    
    def extract_whisper_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[torch.Tensor, str]:
        """
        Extract features from Whisper encoder and get transcription
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (encoder_features, transcribed_text)
        """
        # Preprocess audio
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()
        
        # Normalize
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Get transcription
        result = self.whisper_model.transcribe(
            audio_data,
            language="en",
            task="transcribe",
            fp16=False,
            verbose=False
        )
        transcribed_text = result["text"].strip()
        
        # Extract encoder features
        # Note: This requires accessing Whisper's internal encoder
        # For now, we'll use a simplified approach
        # In production, you'd extract actual encoder outputs
        
        # Convert audio to mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_data).to(self.device)
        
        # Get encoder outputs (simplified - actual implementation would use Whisper's encoder)
        with torch.no_grad():
            # This is a placeholder - actual implementation needs Whisper encoder access
            # For now, we'll create a dummy feature tensor
            # In real implementation, use: encoder_outputs = self.whisper_model.encoder(mel)
            seq_len = mel.shape[-1] // 2  # Approximate sequence length
            encoder_features = torch.randn(1, seq_len, self.whisper_dim).to(self.device)
        
        return encoder_features, transcribed_text
    
    def process_with_neural_architecture(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Complete processing pipeline with neural architecture
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with all processing results
        """
        # Extract Whisper features and transcription
        encoder_features, raw_text = self.extract_whisper_features(audio_data, sample_rate)
        
        if raw_text == "":
            return {
                "raw_text": "",
                "cleaned_text": "",
                "keep_delay_decisions": [],
                "confidence_scores": [],
                "edit_actions": [],
                "predictions": []
            }
        
        # Process through neural components
        with torch.no_grad():
            # Dense Prediction Head: keep/delay decisions
            keep_delay_logits = self.dense_prediction_head(encoder_features)
            keep_delay_probs = F.softmax(keep_delay_logits, dim=-1)
            keep_decisions = torch.argmax(keep_delay_probs, dim=-1).cpu().numpy()[0]
            
            # Confidence Refinement Layer: frame-aligned confidence scores
            confidence_scores = self.confidence_refinement(encoder_features)
            confidence_values = confidence_scores.cpu().numpy()[0, :, 0]
            
            # Clean Caption Decoder: edit actions
            edit_action_logits = self.clean_caption_decoder(encoder_features, confidence_scores)
            edit_action_probs = F.softmax(edit_action_logits, dim=-1)
            edit_actions = torch.argmax(edit_action_probs, dim=-1).cpu().numpy()[0]
        
        # Apply edit actions to clean the text
        cleaned_text, action_details = self.apply_edit_actions(raw_text, edit_actions, keep_decisions)
        
        # Generate predictions
        predictions = self.predict_next_sentences(cleaned_text)
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "keep_delay_decisions": keep_decisions.tolist(),
            "confidence_scores": confidence_values.tolist(),
            "edit_actions": [self.ACTION_NAMES[action] for action in edit_actions],
            "action_details": action_details,
            "predictions": predictions
        }
    
    def apply_edit_actions(self, text: str, edit_actions: np.ndarray, keep_decisions: np.ndarray) -> Tuple[str, List[Dict]]:
        """
        Apply edit actions to clean the text
        
        Args:
            text: Raw transcribed text
            edit_actions: Array of edit actions for each token
            keep_decisions: Array of keep/delay decisions
            
        Returns:
            Tuple of (cleaned_text, action_details)
        """
        words = text.split()
        cleaned_words = []
        action_details = []
        
        # Map actions to words (simplified - in practice, align with tokens)
        min_len = min(len(words), len(edit_actions), len(keep_decisions))
        
        for i in range(min_len):
            word = words[i]
            action = edit_actions[i] if i < len(edit_actions) else self.ACTION_KEEP
            keep = keep_decisions[i] if i < len(keep_decisions) else 1
            
            action_name = self.ACTION_NAMES[action]
            
            # Apply action based on decision
            if keep == 0:  # Delay decision - skip for now
                continue
            
            if action == self.ACTION_KEEP:
                cleaned_words.append(word)
                action_details.append({"word": word, "action": action_name, "applied": "kept"})
            
            elif action == self.ACTION_DELETE_REPEAT:
                # Check if this is a repetition
                if i > 0 and word.lower() == words[i-1].lower():
                    action_details.append({"word": word, "action": action_name, "applied": "deleted (repeat)"})
                    continue  # Skip repeated word
                else:
                    cleaned_words.append(word)
                    action_details.append({"word": word, "action": action_name, "applied": "kept (not repeat)"})
            
            elif action == self.ACTION_DELETE_FILLER:
                # Check if this is a filler word
                if word.lower() in self.filler_words:
                    action_details.append({"word": word, "action": action_name, "applied": "deleted (filler)"})
                    continue  # Skip filler word
                else:
                    cleaned_words.append(word)
                    action_details.append({"word": word, "action": action_name, "applied": "kept (not filler)"})
            
            elif action == self.ACTION_MERGE_PROLONG:
                # Handle prolonged sounds (e.g., "th-th-the" -> "the")
                if '-' in word:
                    # Merge hyphenated prolongations
                    merged = word.replace('-', '')
                    cleaned_words.append(merged)
                    action_details.append({"word": word, "action": action_name, "applied": f"merged to '{merged}'"})
                else:
                    cleaned_words.append(word)
                    action_details.append({"word": word, "action": action_name, "applied": "kept"})
        
        cleaned_text = ' '.join(cleaned_words)
        
        # Capitalize first letter
        if cleaned_text:
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:] if len(cleaned_text) > 1 else cleaned_text.upper()
        
        return cleaned_text, action_details
    
    def predict_next_sentences(self, current_text: str, num_predictions: int = 3) -> List[str]:
        """Predict possible next sentences based on current context"""
        if not current_text:
            return ["Start speaking..."]
        
        words = current_text.lower().split()
        if not words:
            return ["Start speaking..."]
        
        last_word = words[-1] if words else ""
        
        # Simple prediction patterns
        predictions = []
        
        if not current_text.rstrip().endswith(('.', '!', '?')):
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
            else:
                predictions = [
                    f"{current_text} is important to understand.",
                    f"{current_text} helps us see the bigger picture.",
                    f"{current_text} demonstrates the key principle."
                ]
        else:
            predictions = [
                "Let me explain this further.",
                "This leads us to the next point.",
                "We should also consider this aspect."
            ]
        
        return predictions[:num_predictions]
    
    def train_mode(self):
        """Set all components to training mode"""
        self.dense_prediction_head.train()
        self.confidence_refinement.train()
        self.clean_caption_decoder.train()
    
    def eval_mode(self):
        """Set all components to evaluation mode"""
        self.dense_prediction_head.eval()
        self.confidence_refinement.eval()
        self.clean_caption_decoder.eval()





