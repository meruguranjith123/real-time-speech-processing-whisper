# Advanced Neural Architecture for Speech Processing

## Overview

This document describes the advanced neural architecture implemented in `advanced_speech_processor.py` for real-time ASR with stuttering detection and cleaning.

## Architecture Components

### 1. Real-time ASR (Whisper Small Model)

- **Model**: Whisper Small (768M parameters)
- **Purpose**: Base speech recognition with low latency (< 1 second)
- **Output**: Transcribed text with punctuation and word confidence tags
- **Integration**: Preprocessed, clean audio input

### 2. Dense Prediction Head

**Architecture**: Whisper + Bidirectional LSTM

```
Input: Whisper Encoder Features [batch, seq_len, 768]
  ↓
Bidirectional LSTM (2 layers, 256 hidden units)
  ↓
Linear Classifier (512 → 2)
  ↓
Output: Binary Logits [batch, seq_len, 2]
  - [0]: Delay token
  - [1]: Keep token
```

**Purpose**: 
- Determines whether each token should be kept or delayed
- Uses bidirectional context for better decisions
- Processes Whisper encoder outputs

### 3. Confidence Refinement Layer

**Architecture**: 1D CNN + Dense Layer

```
Input: Whisper Features [batch, seq_len, 768]
  ↓
1D CNN (3 layers, 64 filters, kernel_size=3)
  ↓
Dense Layer (64 → 1)
  ↓
Sigmoid Activation
  ↓
Output: Frame-aligned Confidence Scores [batch, seq_len, 1]
```

**Purpose**:
- Produces frame-aligned confidence scores for each token
- Uses 1D convolutions for temporal feature extraction
- Provides confidence information for downstream processing

### 4. Clean Caption Decoder

**Architecture**: Lightweight Edit-Predictor GRU

```
Input: 
  - Whisper Features [batch, seq_len, 768]
  - Confidence Scores [batch, seq_len, 1]
  ↓
Concatenate Features [batch, seq_len, 769]
  ↓
GRU (1 layer, 128 hidden units)
  ↓
Linear Classifier (128 → 4)
  ↓
Output: Edit Action Logits [batch, seq_len, 4]
```

**Edit Actions**:
- **Keep (0)**: Retain the token as-is
- **Delete-Repeat (1)**: Remove repeated token
- **Delete-Filler (2)**: Remove filler word (um, uh, er, etc.)
- **Merge-Prolong (3)**: Merge prolonged sounds (th-th-the → the)

## Complete Pipeline

```
Audio Input
  ↓
Whisper Small Model
  ├─→ Transcription (raw text)
  └─→ Encoder Features [seq_len, 768]
       ↓
       ├─→ Dense Prediction Head
       │    └─→ Keep/Delay Decisions [seq_len]
       │
       ├─→ Confidence Refinement Layer
       │    └─→ Confidence Scores [seq_len, 1]
       │
       └─→ Clean Caption Decoder
            └─→ Edit Actions [seq_len, 4]
                 ↓
                 Apply Actions
                 ↓
                 Cleaned Text Output
```

## Usage

### Basic Usage

```python
from advanced_speech_processor import AdvancedSpeechProcessor
import numpy as np

# Initialize processor
processor = AdvancedSpeechProcessor(
    whisper_model_size="small",
    device="cuda"  # or "cpu"
)

# Process audio
audio_data = np.array([...])  # Your audio data
result = processor.process_with_neural_architecture(audio_data, sample_rate=16000)

print(f"Raw text: {result['raw_text']}")
print(f"Cleaned text: {result['cleaned_text']}")
print(f"Edit actions: {result['edit_actions']}")
print(f"Confidence scores: {result['confidence_scores']}")
```

### Training Mode

The architecture supports fine-tuning:

```python
# Set to training mode
processor.train_mode()

# Fine-tune with your dataset
# (See finetune_advanced.py for training script)
```

## Model Components Details

### Dense Prediction Head

- **Input Dimension**: 768 (Whisper small encoder output)
- **LSTM Hidden Dimension**: 256
- **LSTM Layers**: 2 (bidirectional)
- **Output**: Binary classification (keep/delay)

### Confidence Refinement Layer

- **Input Dimension**: 768
- **CNN Filters**: 64 → 128 → 64
- **Kernel Size**: 3
- **Output**: Confidence scores [0, 1]

### Clean Caption Decoder

- **Input Dimension**: 769 (768 features + 1 confidence)
- **GRU Hidden Dimension**: 128
- **GRU Layers**: 1
- **Output**: 4-class classification (edit actions)

## Integration with Existing System

The advanced architecture can be used alongside or instead of the simple `speech_processor.py`:

1. **Replace**: Use `AdvancedSpeechProcessor` instead of `SpeechProcessor`
2. **Hybrid**: Use both and compare results
3. **Gradual Migration**: Start with simple, migrate to advanced

## Fine-Tuning

To fine-tune the advanced architecture:

1. Use the provided dataset (`finetuning_dataset.json`)
2. Train Dense Prediction Head, Confidence Refinement, and Clean Caption Decoder
3. Keep Whisper encoder frozen (or fine-tune with LoRA)
4. Use edit action labels from dataset

## Performance Characteristics

- **Latency**: < 1 second (with Whisper small)
- **Memory**: ~2GB GPU memory
- **Speed**: Real-time processing capable
- **Accuracy**: Improved stutter detection and cleaning vs. simple approach

## Future Improvements

1. **Better Token Alignment**: Improve alignment between Whisper tokens and edit actions
2. **Attention Mechanisms**: Add attention to Clean Caption Decoder
3. **Multi-task Learning**: Joint training of all components
4. **Quantization**: Reduce model size for deployment
5. **Streaming**: Support streaming audio processing

## References

- Whisper: [OpenAI Whisper](https://github.com/openai/whisper)
- LoRA: Low-Rank Adaptation for efficient fine-tuning
- GRU: Gated Recurrent Unit for sequence modeling

