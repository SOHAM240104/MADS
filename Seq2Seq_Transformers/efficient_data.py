"""
Memory-efficient data loading and processing utilities.
"""

import os
from typing import List, Iterator, Tuple, Optional
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from transformers import PreTrainedTokenizer
import mmap
import logging
from functools import partial
import psutil
from error_handling import DataLoadError, handle_error

logger = logging.getLogger(__name__)

class MemoryEfficientDataset(IterableDataset):
    """Memory-efficient dataset implementation using memory mapping."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 1000,
        max_length: int = 128,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file
            tokenizer: Tokenizer for processing commands
            chunk_size: Number of examples to load at once
            max_length: Maximum sequence length
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        
        # Get file size and estimate number of examples
        self.file_size = os.path.getsize(data_path)
        self.mm = None
        self._estimate_dataset_size()
        
    def _estimate_dataset_size(self) -> None:
        """Estimate the total number of examples in the dataset."""
        try:
            # Memory map the file for efficient reading
            with open(self.data_path, 'r') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
            # Count newlines to estimate number of examples
            self.estimated_size = sum(1 for _ in iter(mm.readline, b''))
            mm.close()
            
        except Exception as e:
            handle_error(DataLoadError("Failed to estimate dataset size", 
                                     {'path': self.data_path}))
    
    def _get_chunk_indices(self, worker_info) -> Iterator[int]:
        """Get indices for the current worker's chunk of data."""
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
        # Calculate chunk size for this worker
        chunk_size = self.estimated_size // num_workers
        start_idx = worker_id * chunk_size
        end_idx = start_idx + chunk_size if worker_id < num_workers - 1 else self.estimated_size
        
        indices = np.arange(start_idx, end_idx)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + worker_id if self.seed else None)
            rng.shuffle(indices)
            
        return iter(indices)
    
    def _read_example(self, index: int) -> Tuple[str, int]:
        """Read a single example from the memory-mapped file."""
        try:
            # Seek to the approximate position
            self.mm.seek(index * 100)  # Approximate average line length
            
            # Read until the next newline
            self.mm.readline()  # Skip partial line
            line = self.mm.readline().decode('utf-8').strip()
            
            if not line:
                raise StopIteration
                
            data = json.loads(line)
            command = data.get('cmd', data.get('command', ''))
            label = 1 if 'docker' in self.data_path else 0
            
            return command, label
            
        except Exception as e:
            handle_error(DataLoadError("Failed to read example", 
                                     {'index': index, 'line': line if 'line' in locals() else None}))
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over the dataset in chunks."""
        worker_info = torch.utils.data.get_worker_info()
        
        try:
            # Memory map the file
            f = open(self.data_path, 'r')
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            indices = self._get_chunk_indices(worker_info)
            current_chunk = []
            
            for idx in indices:
                try:
                    command, label = self._read_example(idx)
                    current_chunk.append((command, label))
                    
                    if len(current_chunk) >= self.chunk_size:
                        yield self._process_chunk(current_chunk)
                        current_chunk = []
                        
                except StopIteration:
                    break
                    
            # Process remaining examples
            if current_chunk:
                yield self._process_chunk(current_chunk)
                
        finally:
            if self.mm is not None:
                self.mm.close()
            f.close()
    
    def _process_chunk(self, chunk: List[Tuple[str, int]]) -> dict:
        """Process a chunk of examples efficiently."""
        commands, labels = zip(*chunk)
        
        # Tokenize the entire chunk at once
        encodings = self.tokenizer(
            list(commands),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def create_efficient_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int,
    max_length: int = 128,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create a memory-efficient dataloader.
    
    Args:
        data_path: Path to the data file
        tokenizer: Tokenizer for processing commands
        batch_size: Batch size
        num_workers: Number of worker processes
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        
    Returns:
        DataLoader instance
    """
    # Calculate optimal chunk size based on available memory
    available_memory = psutil.virtual_memory().available
    estimated_sample_size = max_length * 4  # Rough estimate of bytes per sample
    chunk_size = min(
        batch_size * 100,  # Don't make chunks too large
        max(batch_size, int(available_memory * 0.1 / estimated_sample_size))  # Use 10% of available memory
    )
    
    dataset = MemoryEfficientDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed
    )
    
    return DataLoader(
        dataset,
        batch_size=None,  # Batching is handled by the dataset
        num_workers=num_workers,
        pin_memory=True
    )