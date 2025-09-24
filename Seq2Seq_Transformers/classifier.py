"""
Command Classifier - A transformer-based classifier for bash vs docker commands
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Model and training configuration."""
    
    def __init__(self):
        # Model settings
        self.model_name = "bert-base-uncased"
        self.max_length = 128
        
        # Training settings
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        
        # Data paths
        self.bash_data_path = "/Users/mohamedaamir/Documents/MADS/data/bash_dataset.json"
        self.docker_data_path = "/Users/mohamedaamir/Documents/MADS/data/docker_dataset.json"
        
        # Output paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')

class CommandDataset(Dataset):
    """Dataset for command classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=Config().max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_config_path: str) -> Tuple[List[str], List[int]]:
    """
    Load and combine bash and docker datasets.
    
    Args:
        data_config_path: Path to JSON config containing dataset paths
        
    Returns:
        Tuple of (commands, labels)
    """
    try:
        with open(data_config_path) as f:
            config = json.load(f)
            
        # Load datasets
        with open(config['bash']) as f:
            bash_data = json.load(f)
        with open(config['docker']) as f:
            docker_data = json.load(f)
            
        # Extract commands and assign labels
        commands = []
        labels = []
        
        # Add bash commands (label 0)
        for item in bash_data:
            if isinstance(item, dict) and 'cmd' in item:
                commands.append(item['cmd'])
                labels.append(0)
            elif isinstance(item, str):
                commands.append(item)
                labels.append(0)
                
        # Add docker commands (label 1)
        for item in docker_data:
            if isinstance(item, dict) and 'cmd' in item:
                commands.append(item['cmd'])
                labels.append(1)
            elif isinstance(item, str):
                commands.append(item)
                labels.append(1)
        
        return commands, labels
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def create_data_loaders(
    commands: List[str],
    labels: List[int],
    tokenizer: AutoTokenizer,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Create train, validation, and test data loaders with stratification.
    
    Args:
        commands: List of command strings
        labels: List of corresponding labels
        tokenizer: Transformer tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Calculate class weights for handling imbalance
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.tensor(total_samples / (len(class_counts) * class_counts),
                               dtype=torch.float32)
    
    # Create stratified splits
    train_commands, temp_commands, train_labels, temp_labels = train_test_split(
        commands, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    val_commands, test_commands, val_labels, test_labels = train_test_split(
        temp_commands, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create datasets
    train_dataset = CommandDataset(train_commands, train_labels, tokenizer)
    val_dataset = CommandDataset(val_commands, val_labels, tokenizer)
    test_dataset = CommandDataset(test_commands, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

class CommandClassifier(nn.Module):
    """BERT-based classifier for commands."""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_data() -> Tuple[List[str], List[int]]:
    """Load and combine bash and docker datasets."""
    config = Config()
    
    try:
        # Load bash commands
        with open(config.bash_data_path) as f:
            bash_data = json.load(f)
        
        # Load docker commands
        with open(config.docker_data_path) as f:
            docker_data = json.load(f)
        
        commands = []
        labels = []
        
        # Process bash commands (label 0)
        for item in bash_data:
            if isinstance(item, dict) and 'cmd' in item:
                commands.append(item['cmd'])
                labels.append(0)
            elif isinstance(item, str):
                commands.append(item)
                labels.append(0)
        
        # Process docker commands (label 1)
        for item in docker_data:
            if isinstance(item, dict) and 'cmd' in item:
                commands.append(item['cmd'])
                labels.append(1)
            elif isinstance(item, str):
                commands.append(item)
                labels.append(1)
        
        return commands, labels
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_data_loaders(
    commands: List[str],
    labels: List[int],
    tokenizer: AutoTokenizer,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train, validation, and test data loaders."""
    
    # Calculate class weights for imbalance
    label_counts = np.bincount(labels)
    class_weights = torch.tensor(
        len(labels) / (2 * label_counts),
        dtype=torch.float32
    )
    
    # Create splits
    train_commands, temp_commands, train_labels, temp_labels = train_test_split(
        commands, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    val_commands, test_commands, val_labels, test_labels = train_test_split(
        temp_commands, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create datasets
    train_dataset = CommandDataset(train_commands, train_labels, tokenizer)
    val_dataset = CommandDataset(val_commands, val_labels, tokenizer)
    test_dataset = CommandDataset(test_commands, test_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

def train_model(config: Config) -> CommandClassifier:
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        config: Configuration object
        
    Returns:
        Trained model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        commands, labels = load_data('dataset_config.json')
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            commands, labels, tokenizer, config
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = CommandClassifier(config).to(config.device)
        
        # Initialize optimizer with weight decay
        optimizer = AdamW([
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.0}
        ], lr=config.learning_rate)
        
        # Initialize learning rate scheduler
        total_steps = len(train_loader) * config.num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio
        )
        
        # Initialize loss function with class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(config.device)
        )
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        early_stopping_count = 0
        early_stopping_patience = 3
        
        for epoch in range(config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            # Train epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, config
            )
            
            # Evaluate on validation set
            val_loss, val_metrics = evaluate(model, val_loader, criterion, config)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_count = 0
                
                # Save model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_metrics': val_metrics
                }, config.model_save_path)
                
                # Save tokenizer
                tokenizer.save_pretrained(config.tokenizer_save_path)
            else:
                early_stopping_count += 1
                
            # Early stopping
            if early_stopping_count >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break
        
        # Load best model for testing
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics = evaluate(model, test_loader, criterion, config)
        
        # Log final test metrics
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        # Create confusion matrix
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label']
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu()
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        logger.info("\nConfusion Matrix:")
        logger.info("            Predicted Bash  Predicted Docker")
        logger.info(f"Actual Bash      {cm[0][0]:<14d} {cm[0][1]}")
        logger.info(f"Actual Docker    {cm[1][0]:<14d} {cm[1][1]}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize config
    config = Config()
    
    try:
        # Create dataset_config.json if it doesn't exist
        dataset_config = {
            "bash": "/Users/mohamedaamir/Documents/MADS/data/bash_dataset.json",
            "docker": "/Users/mohamedaamir/Documents/MADS/data/docker_dataset.json"
        }
        
        with open('dataset_config.json', 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        # Train model
        model = train_model(config)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise