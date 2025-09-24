"""
Data validation and preprocessing utilities for command classification.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Metrics for monitoring data quality."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    class_distribution: Dict[str, int]
    avg_sequence_length: float
    max_sequence_length: int
    num_empty_commands: int
    num_malformed_commands: int

class CommandPreprocessor:
    """Handles command string preprocessing and validation."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.invalid_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        self.consecutive_spaces_pattern = re.compile(r'\s+')
        
    def clean_command(self, command: str) -> str:
        """
        Clean and normalize a command string.
        
        Args:
            command: Raw command string
            
        Returns:
            Cleaned command string
        """
        if not isinstance(command, str):
            return ""
            
        # Remove invalid characters
        command = self.invalid_chars_pattern.sub('', command)
        
        # Normalize whitespace
        command = self.consecutive_spaces_pattern.sub(' ', command.strip())
        
        # Remove potentially harmful characters
        command = re.sub(r'[\\`\'"]', '', command)
        
        return command
        
    def validate_command(self, command: str) -> Tuple[bool, str]:
        """
        Validate a command string.
        
        Args:
            command: Command string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command:
            return False, "Empty command"
            
        if len(command) > self.max_length:
            return False, f"Command exceeds maximum length of {self.max_length}"
            
        if self.invalid_chars_pattern.search(command):
            return False, "Command contains invalid characters"
            
        return True, ""

class DatasetValidator:
    """Validates and analyzes dataset quality."""
    
    def __init__(self, preprocessor: CommandPreprocessor):
        self.preprocessor = preprocessor
        
    def validate_dataset(self, 
                        commands: List[str], 
                        labels: List[int]) -> DataQualityMetrics:
        """
        Validate and analyze dataset quality.
        
        Args:
            commands: List of command strings
            labels: List of corresponding labels
            
        Returns:
            DataQualityMetrics object containing quality metrics
        """
        if len(commands) != len(labels):
            raise ValueError("Length mismatch between commands and labels")
            
        valid_samples = 0
        invalid_samples = 0
        total_length = 0
        max_length = 0
        empty_commands = 0
        malformed_commands = 0
        
        valid_commands = []
        valid_labels = []
        
        for command, label in zip(commands, labels):
            is_valid, error = self.preprocessor.validate_command(command)
            
            if is_valid:
                cleaned_command = self.preprocessor.clean_command(command)
                command_length = len(cleaned_command.split())
                
                if command_length == 0:
                    empty_commands += 1
                    invalid_samples += 1
                    continue
                    
                valid_samples += 1
                total_length += command_length
                max_length = max(max_length, command_length)
                
                valid_commands.append(cleaned_command)
                valid_labels.append(label)
            else:
                invalid_samples += 1
                if "invalid characters" in error:
                    malformed_commands += 1
                logger.warning(f"Invalid command: {error}")
                
        class_dist = Counter(valid_labels)
        
        return DataQualityMetrics(
            total_samples=len(commands),
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            class_distribution=dict(class_dist),
            avg_sequence_length=total_length / valid_samples if valid_samples > 0 else 0,
            max_sequence_length=max_length,
            num_empty_commands=empty_commands,
            num_malformed_commands=malformed_commands
        )

def check_data_drift(
    reference_metrics: DataQualityMetrics,
    current_metrics: DataQualityMetrics,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Check for data drift between reference and current datasets.
    
    Args:
        reference_metrics: Metrics from reference dataset
        current_metrics: Metrics from current dataset
        threshold: Maximum allowed drift
        
    Returns:
        Dictionary of drift metrics
    """
    drift_metrics = {}
    
    # Check class distribution drift
    for label in reference_metrics.class_distribution:
        ref_prop = (reference_metrics.class_distribution[label] / 
                   reference_metrics.total_samples)
        curr_prop = (current_metrics.class_distribution[label] / 
                    current_metrics.total_samples)
        drift = abs(ref_prop - curr_prop)
        drift_metrics[f'class_{label}_drift'] = drift
        
        if drift > threshold:
            logger.warning(f"Significant class distribution drift detected for class {label}")
    
    # Check sequence length drift
    length_drift = abs(reference_metrics.avg_sequence_length - 
                      current_metrics.avg_sequence_length)
    drift_metrics['sequence_length_drift'] = length_drift
    
    if length_drift > threshold * reference_metrics.avg_sequence_length:
        logger.warning("Significant sequence length drift detected")
    
    return drift_metrics