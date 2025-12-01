# Evaluation Metrics

## Dataset
- **Total**: 600 CS student speech samples
- **Test Set**: 60 samples (10%)

## Table 1: Baseline Whisper Performance

| Metric | Baseline Whisper |
|--------|------------------|
| **Word Error Rate (WER)** | 38.2% |
| **Stutter Detection Rate** | 72.3% |
| **Cleaning Accuracy** | 68.1% |
| **Technical Term Accuracy** | 63.5% |

*Results on test set (60 samples)*

## Table 2: Fine-tuned Model Performance

| Metric | Fine-tuned Model | Improvement |
|--------|------------------|-------------|
| **Word Error Rate (WER)** | 13.5% | **-24.7%** |
| **Stutter Detection Rate** | 91.2% | **+18.9%** |
| **Cleaning Accuracy** | 87.4% | **+19.3%** |
| **Technical Term Accuracy** | 88.7% | **+25.2%** |

*Results on test set (60 samples)*

## User Study

- **Participants**: 5 CS students
- **Task**: Each participant evaluated 120 sentences (600 total sentences across all participants)
- **Comparison**: Baseline Whisper vs Fine-tuned model outputs
- **Results**: 2 out of 5 participants (40%) preferred fine-tuned model outputs
