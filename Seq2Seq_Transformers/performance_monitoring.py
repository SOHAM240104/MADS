"""
Performance monitoring and profiling utilities.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import torch
import numpy as np
from torch.cuda import nvtx
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for various performance metrics."""
    
    batch_time: float
    data_loading_time: float
    forward_time: float
    backward_time: float
    optimization_time: float
    gpu_memory_used: Optional[float]
    cpu_memory_used: float
    gpu_utilization: Optional[float]
    throughput: float  # samples per second

class PerformanceMonitor:
    """Monitors and logs performance metrics during training."""
    
    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics_history = defaultdict(list)
        self.current_batch_metrics = {}
        self.epoch_metrics = {}
        
        self.start_time = None
        self.batch_start_time = None
        
    @contextmanager
    def measure_time(self, metric_name: str):
        """Context manager to measure execution time of a code block."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            self.current_batch_metrics[f"{metric_name}_time"] = elapsed_time
    
    def start_epoch(self, epoch: int):
        """Start monitoring a new epoch."""
        self.start_time = time.time()
        self.epoch_metrics = defaultdict(list)
        self.current_epoch = epoch
        
    def start_batch(self):
        """Start monitoring a new batch."""
        self.batch_start_time = time.time()
        self.current_batch_metrics = {}
        
    def end_batch(self, batch_size: int):
        """
        End batch monitoring and collect metrics.
        
        Args:
            batch_size: Number of samples in the batch
        """
        batch_time = time.time() - self.batch_start_time
        
        # Collect GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_memory_used = None
            
        # Collect CPU metrics
        cpu_memory_used = psutil.Process().memory_info().rss / 1024**2  # MB
        
        # Calculate throughput
        throughput = batch_size / batch_time
        
        metrics = PerformanceMetrics(
            batch_time=batch_time,
            data_loading_time=self.current_batch_metrics.get('data_loading_time', 0),
            forward_time=self.current_batch_metrics.get('forward_time', 0),
            backward_time=self.current_batch_metrics.get('backward_time', 0),
            optimization_time=self.current_batch_metrics.get('optimization_time', 0),
            gpu_memory_used=gpu_memory_used,
            cpu_memory_used=cpu_memory_used,
            gpu_utilization=self._get_gpu_utilization(),
            throughput=throughput
        )
        
        # Store metrics
        for key, value in metrics.__dict__.items():
            self.epoch_metrics[key].append(value)
        
    def end_epoch(self):
        """End epoch monitoring and compute summary statistics."""
        epoch_time = time.time() - self.start_time
        
        # Compute summary statistics
        summary = {
            'epoch': self.current_epoch,
            'total_time': epoch_time
        }
        
        # Add averages for all metrics
        for key, values in self.epoch_metrics.items():
            if values:  # Check if we have any values
                values = [v for v in values if v is not None]  # Filter out None values
                if values:  # Check if we still have values after filtering
                    summary[f'avg_{key}'] = np.mean(values)
                    summary[f'std_{key}'] = np.std(values)
                    summary[f'min_{key}'] = np.min(values)
                    summary[f'max_{key}'] = np.max(values)
        
        # Store summary
        self.metrics_history['epoch_summaries'].append(summary)
        
        # Log summary
        logger.info(f"Epoch {self.current_epoch} Performance Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Save metrics to file
        self._save_metrics()
        
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage if available."""
        try:
            if torch.cuda.is_available():
                # Note: This is a placeholder. In a real implementation,
                # you would use nvidia-smi or a similar tool to get actual GPU utilization
                return None
        except:
            return None
    
    def _save_metrics(self):
        """Save performance metrics to a JSON file."""
        metrics_file = self.log_dir / f"metrics_epoch_{self.current_epoch}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected performance metrics."""
        summary = {
            'total_epochs': len(self.metrics_history['epoch_summaries']),
            'average_epoch_time': np.mean([
                summary['total_time'] 
                for summary in self.metrics_history['epoch_summaries']
            ]),
            'average_throughput': np.mean([
                summary['avg_throughput'] 
                for summary in self.metrics_history['epoch_summaries']
            ]),
            'peak_memory_usage': {
                'cpu': max(summary['max_cpu_memory_used'] 
                          for summary in self.metrics_history['epoch_summaries']),
                'gpu': max(summary['max_gpu_memory_used'] 
                          for summary in self.metrics_history['epoch_summaries']
                          if 'max_gpu_memory_used' in summary)
                if any('max_gpu_memory_used' in s 
                      for s in self.metrics_history['epoch_summaries']) else None
            }
        }
        
        return summary

@contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX profiling range."""
    try:
        if torch.cuda.is_available():
            nvtx.range_push(name)
        yield
    finally:
        if torch.cuda.is_available():
            nvtx.range_pop()