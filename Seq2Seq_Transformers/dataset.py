"""
Dataset and DataLoader implementations for command generation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommandDataset(Dataset):
    """Dataset for command generation with balanced sampling"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.max_length
        self.data = []
        
        # Load bash data
        logger.info(f"Loading bash data from {config.bash_data}")
        with open(config.bash_data) as f:
            bash_data = json.load(f)
            if 'examples' in bash_data:
                for item in bash_data['examples']:
                    if isinstance(item, dict) and 'input' in item and 'output' in item:
                        self.data.append({
                            'input': item['input'],
                            'target': item['output'],
                            'type': 'bash'
                        })
        
        # Load docker data
        logger.info(f"Loading docker data from {config.docker_data}")
        with open(config.docker_data) as f:
            docker_data = json.load(f)
            if 'examples' in docker_data:
                for item in docker_data['examples']:
                    if isinstance(item, dict) and 'input' in item and 'output' in item:
                        self.data.append({
                            'input': item['input'],
                            'target': item['output'],
                            'type': 'docker'
                        })
        
        logger.info(f"Loaded {len(self.data)} examples")
        logger.info(f"Bash commands: {sum(1 for x in self.data if x['type'] == 'bash')}")
        logger.info(f"Docker commands: {sum(1 for x in self.data if x['type'] == 'docker')}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Prepare input sequence
        input_text = str(item['input'])  # Ensure input is string
        if not input_text.endswith('.'):
            input_text += '.'
            
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (with special tokens)
        target_text = f"{config.tokenizer.cls_token} {item['target']} {config.tokenizer.sep_token}"
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'type': item['type']
        }


def create_dataloaders(
    tokenizer: AutoTokenizer,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with balanced sampling"""
    
    batch_size = batch_size or config.batch_size
    dataset = CommandDataset(tokenizer)
    
    # Calculate weights for balanced sampling
    bash_count = sum(1 for x in dataset.data if x['type'] == 'bash')
    docker_count = sum(1 for x in dataset.data if x['type'] == 'docker')
    
    weights = []
    bash_weight = 1.0 / bash_count if bash_count > 0 else 0
    docker_weight = 1.0 / docker_count if docker_count > 0 else 0
    
    for item in dataset.data:
        weight = docker_weight if item['type'] == 'docker' else bash_weight
        weights.append(weight)
    
    # Create indices for splits
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()
    
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create samplers
    train_sampler = WeightedRandomSampler(
        weights=[weights[i] for i in train_indices],
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader