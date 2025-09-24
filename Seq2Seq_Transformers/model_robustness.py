"""
Model robustness enhancements and data augmentation techniques.
"""

import random
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedTokenizer

class RobustTokenizer:
    """Wrapper around tokenizer with additional robustness features."""
    
    def __init__(self, base_tokenizer: PreTrainedTokenizer):
        self.tokenizer = base_tokenizer
        self.special_tokens = {
            'unk_token': self.tokenizer.unk_token,
            'pad_token': self.tokenizer.pad_token,
            'cls_token': self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else None,
            'sep_token': self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else None
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove repeated punctuation
        text = re.sub(r'([.,!?]){2,}', r'\1', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        return text
    
    def augment_text(self, text: str, 
                    p_typo: float = 0.1,
                    p_drop: float = 0.1) -> str:
        """Apply text augmentation techniques."""
        words = text.split()
        augmented_words = []
        
        for word in words:
            if random.random() < p_drop:
                continue  # Drop word
                
            if random.random() < p_typo:
                word = self._introduce_typo(word)
                
            augmented_words.append(word)
            
        return ' '.join(augmented_words)
    
    def _introduce_typo(self, word: str) -> str:
        """Introduce a random typo in the word."""
        if len(word) < 2:
            return word
            
        typo_type = random.choice(['swap', 'delete', 'insert', 'replace'])
        
        if typo_type == 'swap':
            idx = random.randint(0, len(word)-2)
            word_list = list(word)
            word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
            return ''.join(word_list)
            
        elif typo_type == 'delete':
            idx = random.randint(0, len(word)-1)
            return word[:idx] + word[idx+1:]
            
        elif typo_type == 'insert':
            idx = random.randint(0, len(word))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return word[:idx] + char + word[idx:]
            
        else:  # replace
            idx = random.randint(0, len(word)-1)
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return word[:idx] + char + word[idx+1:]

class MixupLayer(nn.Module):
    """Implements Mixup augmentation for better regularization."""
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup to a batch of inputs and labels.
        
        Args:
            x: Input tensor of shape [batch_size, ...]
            y: Label tensor of shape [batch_size]
            
        Returns:
            Tuple of (mixed inputs, mixed labels)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * F.one_hot(y, num_classes=2) + \
                 (1 - lam) * F.one_hot(y[index], num_classes=2)
        
        return mixed_x, mixed_y

class LabelSmoothing(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class RobustCommandClassifier(nn.Module):
    """Enhanced command classifier with robustness features."""
    
    def __init__(self, base_model: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        
        # Additional robustness features
        self.mixup = MixupLayer(alpha=0.2)
        self.label_smoothing = LabelSmoothing(smoothing=0.1)
        
        # Stochastic depth
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout(p=dropout * (i+1)/len(base_model.transformer.layers))
            for i in range(len(base_model.transformer.layers))
        ])
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                use_mixup: bool = True) -> torch.Tensor:
        """Forward pass with robustness features."""
        
        # Apply mixup during training if enabled
        if self.training and use_mixup and labels is not None:
            input_ids, mixed_labels = self.mixup(input_ids, labels)
        
        # Forward through base model with stochastic depth
        for i, layer in enumerate(self.base_model.transformer.layers):
            x = layer(input_ids, attention_mask)
            if self.training:
                x = self.layer_dropouts[i](x)
        
        # Get logits
        logits = self.base_model.classifier(x)
        
        if self.training and use_mixup and labels is not None:
            loss = self.label_smoothing(logits, mixed_labels)
            return logits, loss
        
        return logits

def create_robust_model(base_model: nn.Module, 
                       tokenizer: PreTrainedTokenizer) -> Tuple[nn.Module, RobustTokenizer]:
    """Create a robust model with enhanced tokenizer."""
    robust_model = RobustCommandClassifier(base_model)
    robust_tokenizer = RobustTokenizer(tokenizer)
    
    return robust_model, robust_tokenizer