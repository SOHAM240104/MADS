"""
Simple and effective command classifier using BERT
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
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # AdamW is now in torch.optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.max_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.num_epochs = 10
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        self.data_dir = Path("/Users/mohamedaamir/Documents/MADS/data")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')

class CommandDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CommandClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

def load_dataset(config: Config) -> Tuple[List[str], List[int]]:
    """Load and combine bash and docker datasets."""
    try:
        # Load datasets
        with open(config.data_dir / "bash_dataset.json") as f:
            bash_data = json.load(f)
        with open(config.data_dir / "docker_dataset.json") as f:
            docker_data = json.load(f)
        
        commands = []
        labels = []
        
        # Process bash commands (label 0)
        if 'examples' in bash_data:
            for item in bash_data['examples']:
                if isinstance(item, dict):
                    # Use both input (description) and output (command)
                    if 'input' in item and 'output' in item:
                        text = f"{item['input']} {item['output']}"
                        commands.append(text)
                        labels.append(0)
        else:
            logger.warning("No examples found in bash dataset")
        
        # Process docker commands (label 1)
        if 'examples' in docker_data:
            for item in docker_data['examples']:
                if isinstance(item, dict):
                    # Use both input (description) and output (command)
                    if 'input' in item and 'output' in item:
                        text = f"{item['input']} {item['output']}"
                        commands.append(text)
                        labels.append(1)
        else:
            logger.warning("No examples found in docker dataset")
        
        if not commands:
            raise ValueError("No valid commands found in datasets")
            
        logger.info(f"Loaded {len(commands)} commands ({labels.count(0)} bash, {labels.count(1)} docker)")
        return commands, labels
    
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def create_dataloaders(texts: List[str], labels: List[int], config: Config, tokenizer: AutoTokenizer):
    """Create train, validation, and test data loaders."""
    
    # Calculate class weights for imbalance
    counts = np.bincount(labels)
    class_weights = torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)
    
    # Check minimum class count
    min_count = min(counts)
    if min_count < 2:
        logger.warning(f"Found class with only {min_count} instance(s). Switching to random split.")
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
    else:
        # Create stratified splits
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, stratify=labels, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
    
    # Create datasets
    train_dataset = CommandDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = CommandDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset = CommandDataset(test_texts, test_labels, tokenizer, config.max_length)
    
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

def train(config: Config):
    """Main training function."""
    
    try:
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load data
        logger.info("Loading datasets...")
        texts, labels = load_dataset(config)
        
        # Initialize tokenizer and model
        logger.info("Initializing model...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = CommandClassifier(config.model_name).to(config.device)
        
        # Create dataloaders
        train_loader, val_loader, test_loader, class_weights = create_dataloaders(
            texts, labels, config, tokenizer
        )
        
        # Setup training
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        num_training_steps = len(train_loader) * config.num_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(config.device)
        )
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 3
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            # Training
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(config.device)
                    attention_mask = batch['attention_mask'].to(config.device)
                    labels = batch['labels'].to(config.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100. * correct / total
            
            logger.info(f"Epoch {epoch + 1}: ")
            logger.info(f"  Training Loss: {avg_train_loss:.4f}")
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"  Validation Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'val_accuracy': best_val_acc
                }, config.model_dir / 'best_model.pt')
                
                # Save tokenizer
                tokenizer.save_pretrained(config.model_dir)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        # Test evaluation
        logger.info("\nEvaluating on test set...")
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100. * correct / total
        logger.info(f"Test Loss: {test_loss/len(test_loader):.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Save final metrics
        metrics = {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_acc,
            'best_val_loss': best_val_loss,
            'final_epoch': config.num_epochs
        }
        
        with open(config.model_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    config = Config()
    train(config)