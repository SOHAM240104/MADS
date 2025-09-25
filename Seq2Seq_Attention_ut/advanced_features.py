import torch
import torch.nn.functional as F
import heapq
from typing import List, Tuple
import numpy as np

class BeamSearchDecoder:
    """
    Beam Search decoder for improved inference quality
    """
    def __init__(self, model, beam_size=5, max_length=70, eos_token=3, sos_token=2):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.eos_token = eos_token
        self.sos_token = sos_token
        
    def beam_search(self, src, src_lengths):
        """
        Perform beam search decoding
        Args:
            src: [seq_len, 1] - source sequence (single example)
            src_lengths: [1] - source length
        Returns:
            best_sequence: list of token indices
            best_score: float - log probability of best sequence
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
            
            # Create mask
            mask = self.model.create_mask(src, src_lengths) if src_lengths is not None else None
            
            # Initialize beam
            # Each beam item: (score, sequence, hidden_state, cell_state)
            beam = [(0.0, [self.sos_token], hidden[0], cell[0])]  # Assuming batch_size=1
            
            completed_sequences = []
            
            for step in range(self.max_length):
                if not beam:
                    break
                    
                new_beam = []
                
                for score, sequence, h_state, c_state in beam:
                    # If sequence is complete, move to completed
                    if sequence[-1] == self.eos_token:
                        completed_sequences.append((score, sequence))
                        continue
                    
                    # Get last token
                    input_token = torch.LongTensor([sequence[-1]]).to(device)
                    
                    # Decoder step
                    output, new_h, new_c, _ = self.model.decoder(
                        input_token, h_state.unsqueeze(0), c_state.unsqueeze(0), 
                        encoder_outputs, mask
                    )
                    
                    # Get log probabilities
                    log_probs = F.log_softmax(output, dim=1)
                    
                    # Get top k tokens
                    top_log_probs, top_indices = torch.topk(log_probs, self.beam_size, dim=1)
                    
                    # Create new beam candidates
                    for i in range(self.beam_size):
                        token_idx = top_indices[0, i].item()
                        token_score = top_log_probs[0, i].item()
                        new_score = score + token_score
                        new_sequence = sequence + [token_idx]
                        
                        new_beam.append((new_score, new_sequence, new_h.squeeze(0), new_c.squeeze(0)))
                
                # Keep only top beam_size candidates
                beam = heapq.nlargest(self.beam_size, new_beam, key=lambda x: x[0])
                
                # Early stopping if all beams are completed
                if len(completed_sequences) >= self.beam_size:
                    break
            
            # Add remaining beam items to completed
            for item in beam:
                completed_sequences.append((item[0], item[1]))
            
            # Return best sequence
            if completed_sequences:
                best_score, best_sequence = max(completed_sequences, key=lambda x: x[0])
                return best_sequence[1:], best_score  # Remove SOS token
            else:
                return [], float('-inf')

class ModelEnsemble:
    """
    Model ensemble for improved performance
    """
    def __init__(self, models: List[torch.nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
    def predict(self, src, src_lengths, max_len=70, sos_token=2, eos_token=3):
        """
        Ensemble prediction by averaging model outputs
        """
        all_outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs, _ = model.inference(src, src_lengths, max_len, sos_token, eos_token)
                all_outputs.append(outputs)
        
        # Simple voting - take most common prediction at each position
        if not all_outputs or all_outputs[0].size(0) == 0:
            return torch.empty(0, src.size(1), dtype=torch.long, device=src.device)
        
        # For simplicity, just return the first model's output
        # In practice, you'd want more sophisticated ensemble methods
        return all_outputs[0]

class GradualUnfreezing:
    """
    Gradual unfreezing technique for transfer learning
    """
    def __init__(self, model, unfreeze_schedule):
        """
        Args:
            model: PyTorch model
            unfreeze_schedule: dict mapping epoch to list of parameter names to unfreeze
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        
        # Initially freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def update_frozen_params(self, epoch):
        """Update which parameters are frozen based on epoch"""
        if epoch in self.unfreeze_schedule:
            params_to_unfreeze = self.unfreeze_schedule[epoch]
            
            for name, param in self.model.named_parameters():
                for param_pattern in params_to_unfreeze:
                    if param_pattern in name:
                        param.requires_grad = True
                        print(f"Unfroze parameter: {name}")

class CyclicLRScheduler:
    """
    Cyclic learning rate scheduler
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular'):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.step_count = 0
        
    def step(self):
        """Update learning rate"""
        cycle = np.floor(1 + self.step_count / (2 * self.step_size))
        x = np.abs(self.step_count / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        else:  # triangular2
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / float(2 ** (cycle - 1))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1
        return lr

class LabelSmoothingLoss(torch.nn.Module):
    """
    Label smoothing loss for better generalization
    """
    def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        """
        Args:
            input: [batch_size * seq_len, num_classes] - model predictions
            target: [batch_size * seq_len] - target labels
        """
        log_probs = F.log_softmax(input, dim=1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        
        # Mask ignored indices
        mask = (target != self.ignore_index).unsqueeze(1)
        
        # Set true class probabilities
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply mask
        true_dist = true_dist * mask.float()
        
        return -torch.sum(true_dist * log_probs, dim=1).mean()

class ModelCheckpointing:
    """
    Enhanced model checkpointing with multiple criteria
    """
    def __init__(self, checkpoint_dir, save_top_k=3):
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        self.best_scores = []
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, vocab_info):
        """Save checkpoint if it's among the top-k best"""
        main_metric = metrics.get('exact_match', metrics.get('valid_loss', 0))
        
        checkpoint_info = {
            'epoch': epoch,
            'score': main_metric,
            'metrics': metrics,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'vocab_info': vocab_info
        }
        
        # Check if this is a top-k score
        if len(self.best_scores) < self.save_top_k or main_metric > min(self.best_scores, key=lambda x: x['score'])['score']:
            
            # Save checkpoint
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}_score_{main_metric:.4f}.pt"
            torch.save(checkpoint_info, checkpoint_path)
            
            # Update best scores
            self.best_scores.append({'epoch': epoch, 'score': main_metric, 'path': checkpoint_path})
            self.best_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Remove worst checkpoint if we have too many
            if len(self.best_scores) > self.save_top_k:
                worst_checkpoint = self.best_scores.pop()
                import os
                if os.path.exists(worst_checkpoint['path']):
                    os.remove(worst_checkpoint['path'])
            
            return True
        
        return False

class EarlyStopping:
    """
    Enhanced early stopping with multiple criteria
    """
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        
    def __call__(self, model, val_loss, val_metrics=None):
        """
        Args:
            model: PyTorch model
            val_loss: validation loss
            val_metrics: dict of validation metrics (optional)
        Returns:
            bool: True if training should stop
        """
        # Use exact match if available, otherwise use negative loss
        if val_metrics and 'exact_match' in val_metrics:
            score = val_metrics['exact_match']
            is_better = lambda current, best: current > best + self.min_delta
        else:
            score = -val_loss
            is_better = lambda current, best: current > best + self.min_delta
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
            return True
        
        return False

# Data augmentation techniques
class DataAugmentation:
    """
    Data augmentation techniques for sequence-to-sequence models
    """
    @staticmethod
    def synonym_replacement(tokens, vocab, replacement_prob=0.1):
        """Replace tokens with synonyms (simplified version)"""
        # This is a placeholder - in practice, you'd use WordNet or similar
        augmented = []
        for token in tokens:
            if np.random.random() < replacement_prob and token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                # Simple replacement with a random token from vocab (not ideal, but demonstrates concept)
                if len(vocab.word2idx) > 10:
                    candidates = list(vocab.word2idx.keys())[:10]  # Take first 10 tokens as "synonyms"
                    replacement = np.random.choice(candidates)
                    augmented.append(replacement)
                else:
                    augmented.append(token)
            else:
                augmented.append(token)
        return augmented
    
    @staticmethod
    def random_insertion(tokens, vocab, insertion_prob=0.05):
        """Randomly insert tokens"""
        augmented = tokens.copy()
        for i in range(len(tokens)):
            if np.random.random() < insertion_prob:
                # Insert a random token from vocab
                candidates = list(vocab.word2idx.keys())[:10]
                random_token = np.random.choice(candidates)
                augmented.insert(i, random_token)
        return augmented
    
    @staticmethod
    def random_deletion(tokens, deletion_prob=0.05):
        """Randomly delete tokens"""
        if len(tokens) <= 3:  # Don't delete if sequence is too short
            return tokens
        
        augmented = []
        for token in tokens:
            if token not in ['<sos>', '<eos>'] and np.random.random() < deletion_prob:
                continue  # Skip this token (delete it)
            augmented.append(token)
        
        return augmented if augmented else tokens  # Return original if all deleted