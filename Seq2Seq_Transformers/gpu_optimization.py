"""
GPU optimization and efficient batch processing utilities.
"""

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Handles GPU memory optimization and efficient batch processing."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_gradient_norm: float = 1.0
    ):
        """
        Initialize GPU optimizer.
        
        Args:
            model: The model to optimize
            device: Device to use
            use_mixed_precision: Whether to use automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_gradient_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_gradient_norm = max_gradient_norm
        
        # Initialize mixed precision training
        self.scaler = amp.GradScaler() if self.use_mixed_precision else None
        
        # Enable cudnn benchmarking for better performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Initialize memory trackers
        self.peak_memory = 0
        self.current_memory = 0
        
    def optimize_memory(self):
        """Apply memory optimizations."""
        if self.device.type == 'cuda':
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Enable memory efficient optimizations
            torch.backends.cudnn.deterministic = False
            
            # Use channels last memory format for better performance
            self.model = self.model.to(memory_format=torch.channels_last)
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.use_mixed_precision:
            with amp.autocast():
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Perform backward pass with gradient accumulation and clipping.
        
        Args:
            loss: The loss tensor
            optimizer: The optimizer
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        if self.use_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Update on gradient accumulation steps
        if (self.model.training_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(optimizer)
                
            # Clip gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_gradient_norm
            )
            
            if self.use_mixed_precision:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad()
    
    def prefetch_next_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Prefetch next batch to GPU memory.
        
        Args:
            batch: Dictionary containing the next batch tensors
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
    
    def track_memory(self) -> Dict[str, float]:
        """
        Track GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.device.type == 'cuda':
            current = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            
            self.current_memory = current
            self.peak_memory = max(self.peak_memory, peak)
            
            return {
                'current_memory_mb': current,
                'peak_memory_mb': self.peak_memory
            }
        return {}

def setup_distributed_training(
    model: nn.Module,
    device: torch.device
) -> Optional[DistributedDataParallel]:
    """
    Setup distributed training if multiple GPUs are available.
    
    Args:
        model: The model to distribute
        device: The device to use
        
    Returns:
        Distributed model if multiple GPUs available, else None
    """
    if device.type != 'cuda':
        return None
        
    if torch.cuda.device_count() > 1:
        # Initialize distributed process group
        dist.init_process_group(backend='nccl')
        
        # Wrap model in DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device
        )
        
        return model
    return None

class BatchProcessor:
    """Efficient batch processing with prefetching and memory optimization."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu_optimizer: GPUOptimizer,
        batch_size: int
    ):
        """
        Initialize batch processor.
        
        Args:
            model: The model to process batches
            optimizer: The optimizer
            gpu_optimizer: GPU optimization handler
            batch_size: Batch size
        """
        self.model = model
        self.optimizer = optimizer
        self.gpu_optimizer = gpu_optimizer
        self.batch_size = batch_size
        
        # Initialize prefetch queue
        self.next_batch = None
        
    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Process a single batch efficiently.
        
        Args:
            batch: The current batch
            is_training: Whether in training mode
            
        Returns:
            Tuple of (predictions, loss if training)
        """
        # Move batch to GPU efficiently
        self.gpu_optimizer.prefetch_next_batch(batch)
        
        # Track memory usage
        memory_stats = self.gpu_optimizer.track_memory()
        if memory_stats:
            logger.debug(f"Memory usage: {memory_stats}")
        
        with torch.set_grad_enabled(is_training):
            with self.gpu_optimizer.autocast():
                # Forward pass
                outputs = self.model(**batch)
                
                if is_training:
                    loss = outputs.loss
                    # Backward pass with optimization
                    self.gpu_optimizer.backward(loss, self.optimizer)
                    return outputs.logits, loss.item()
                
                return outputs.logits, None
    
    @torch.no_grad()
    def prefetch_next(self, next_batch: Optional[Dict[str, torch.Tensor]]):
        """
        Prefetch next batch to GPU memory.
        
        Args:
            next_batch: The next batch to prefetch
        """
        if next_batch is not None:
            self.next_batch = {
                k: v.to(self.gpu_optimizer.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in next_batch.items()
            }