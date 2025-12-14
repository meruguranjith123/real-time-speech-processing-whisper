#!/usr/bin/env python3
"""
Fine-tuning script for Whisper model on CS student stuttering speech
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning

This script fine-tunes the Whisper model using the provided dataset
to improve transcription accuracy for stuttered speech.
"""

import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Try to import required libraries
try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        WhisperTokenizer,
        WhisperFeatureExtractor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        EarlyStoppingCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers and peft libraries not found.")
    print("Install with: pip install transformers peft accelerate")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StutteringDataset(Dataset):
    """Dataset class for stuttering speech recognition"""
    
    def __init__(self, data: List[Dict], processor, is_training: bool = True):
        """
        Initialize dataset
        
        Args:
            data: List of training samples with 'raw_transcription' and 'cleaned_transcription'
            processor: WhisperProcessor for tokenization
            is_training: Whether this is training data
        """
        self.data = data
        self.processor = processor
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # For text-to-text fine-tuning, we use raw as input and cleaned as target
        # In a real scenario, you would have audio files here
        raw_text = sample['raw_transcription']
        cleaned_text = sample['cleaned_transcription']
        
        # Tokenize inputs (raw transcription)
        inputs = self.processor(
            text=raw_text,
            return_tensors="pt",
            padding="max_length",
            max_length=448,  # Whisper max length
            truncation=True
        )
        
        # Tokenize labels (cleaned transcription)
        labels = self.processor.tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding="max_length",
            max_length=448,
            truncation=True
        ).input_ids
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze(),
            'raw_text': raw_text,
            'cleaned_text': cleaned_text
        }


def load_dataset(dataset_path: str) -> Tuple[List[Dict], Dict]:
    """
    Load dataset from JSON file
    
    Args:
        dataset_path: Path to finetuning_dataset.json
        
    Returns:
        Tuple of (training_samples, dataset_info)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('training_samples', [])
    dataset_info = data.get('dataset_info', {})
    config = data.get('fine_tuning_config', {})
    
    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"Dataset: {dataset_info.get('name', 'Unknown')}")
    
    return samples, dataset_info, config


def create_lora_config(config: Dict) -> LoraConfig:
    """
    Create LoRA configuration
    
    Args:
        config: Fine-tuning configuration from dataset
        
    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def prepare_model_and_processor(model_name: str = "openai/whisper-base", device: str = "auto"):
    """
    Prepare Whisper model and processor
    
    Args:
        model_name: HuggingFace model name
        device: Device to use (auto, cuda, cpu)
        
    Returns:
        Tuple of (model, processor)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model: {model_name} on {device}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    logger.info("Model and processor loaded successfully")
    
    return model, processor


def train(
    model,
    processor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    config: Dict,
    dataset_info: Dict
):
    """
    Train the model using LoRA fine-tuning
    
    Args:
        model: Whisper model
        processor: Whisper processor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save the model
        config: Fine-tuning configuration
        dataset_info: Dataset information
    """
    logger.info("Setting up LoRA for fine-tuning")
    
    # Create LoRA config
    lora_config = create_lora_config(config)
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get('batch_size', 4),
        per_device_eval_batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=1,
        learning_rate=float(config.get('learning_rate', 1e-5)),
        num_train_epochs=config.get('epochs', 10),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_steps=50,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use FP16 if CUDA available
        report_to="none",
        push_to_hub=False,
    )
    
    # Data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        processor=processor,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    logger.info("Starting training...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Train
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "dataset_info": dataset_info,
        "training_config": config,
        "training_args": training_args.to_dict(),
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info("Training completed!")


def split_dataset(samples: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        samples: List of all samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    np.random.seed(42)
    np.random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    logger.info(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for stuttering speech")
    parser.add_argument(
        "--dataset",
        type=str,
        default="finetuning_dataset.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-base",
        help="Base Whisper model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./whisper-finetuned",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of training samples"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of validation samples"
    )
    
    args = parser.parse_args()
    
    # Check if required libraries are available
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Required libraries not found!")
        logger.error("Please install: pip install transformers peft accelerate datasets")
        return
    
    # Load dataset
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        return
    
    samples, dataset_info, config = load_dataset(args.dataset)
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # Prepare model and processor
    model, processor = prepare_model_and_processor(
        model_name=args.model,
        device=args.device
    )
    
    # Create datasets
    train_dataset = StutteringDataset(train_samples, processor, is_training=True)
    val_dataset = StutteringDataset(val_samples, processor, is_training=False)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        config=config,
        dataset_info=dataset_info
    )
    
    logger.info("=" * 50)
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()





