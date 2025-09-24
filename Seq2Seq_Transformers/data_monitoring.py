"""
Data pipeline monitoring and quality control utilities.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
import pandas as pd
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class DataStats:
    """Statistics about the dataset."""
    
    num_samples: int
    class_distribution: Dict[str, int]
    avg_sequence_length: float
    num_unique_tokens: int
    token_frequency: Dict[str, int]
    common_patterns: List[str]
    outliers: List[int]
    timestamp: str

class DataMonitor:
    """Monitors data quality and distribution shifts."""
    
    def __init__(self, 
                 save_dir: str = "data_monitoring",
                 reference_stats: Optional[Dict[str, Any]] = None):
        """
        Initialize data monitor.
        
        Args:
            save_dir: Directory to save monitoring results
            reference_stats: Reference statistics to compare against
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.reference_stats = reference_stats
        self.history = defaultdict(list)
        
    def compute_stats(self, 
                     texts: List[str],
                     labels: List[int]) -> DataStats:
        """
        Compute comprehensive statistics for the dataset.
        
        Args:
            texts: List of command strings
            labels: List of corresponding labels
            
        Returns:
            DataStats object with computed statistics
        """
        # Basic statistics
        num_samples = len(texts)
        class_dist = {str(i): sum(1 for l in labels if l == i) 
                     for i in set(labels)}
        
        # Sequence length statistics
        seq_lengths = [len(text.split()) for text in texts]
        avg_seq_length = np.mean(seq_lengths)
        
        # Token statistics
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        unique_tokens = set(all_tokens)
        token_freq = defaultdict(int)
        for token in all_tokens:
            token_freq[token] += 1
            
        # Find common patterns using n-grams
        common_patterns = self._find_common_patterns(texts)
        
        # Detect outliers in sequence lengths
        outliers = self._detect_outliers(seq_lengths)
        
        return DataStats(
            num_samples=num_samples,
            class_distribution=class_dist,
            avg_sequence_length=avg_seq_length,
            num_unique_tokens=len(unique_tokens),
            token_frequency=dict(token_freq),
            common_patterns=common_patterns,
            outliers=outliers,
            timestamp=datetime.now().isoformat()
        )
    
    def _find_common_patterns(self, texts: List[str], 
                            n_gram_size: int = 3,
                            top_k: int = 10) -> List[str]:
        """Find common n-gram patterns in the texts."""
        n_grams = defaultdict(int)
        
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n_gram_size + 1):
                n_gram = ' '.join(tokens[i:i + n_gram_size])
                n_grams[n_gram] += 1
                
        # Sort by frequency
        sorted_patterns = sorted(
            n_grams.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [pattern for pattern, _ in sorted_patterns[:top_k]]
    
    def _detect_outliers(self, values: List[float],
                        threshold: float = 3.0) -> List[int]:
        """
        Detect outliers using z-score method.
        
        Args:
            values: List of values to check for outliers
            threshold: Z-score threshold for outlier detection
            
        Returns:
            List of indices of detected outliers
        """
        z_scores = np.abs(stats.zscore(values))
        return [i for i, z in enumerate(z_scores) if z > threshold]
    
    def check_distribution_shift(self,
                               current_stats: DataStats,
                               significance_level: float = 0.05) -> Dict[str, float]:
        """
        Check for distribution shifts between current and reference statistics.
        
        Args:
            current_stats: Current dataset statistics
            significance_level: P-value threshold for significance
            
        Returns:
            Dictionary of shift detection results
        """
        if not self.reference_stats:
            logger.warning("No reference statistics available for comparison")
            return {}
            
        shifts = {}
        
        # Check class distribution shift
        chi2, p_value = stats.chisquare(
            list(current_stats.class_distribution.values()),
            list(self.reference_stats['class_distribution'].values())
        )
        shifts['class_distribution_shift'] = {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'is_significant': p_value < significance_level
        }
        
        # Check sequence length distribution
        t_stat, p_value = stats.ttest_ind(
            current_stats.avg_sequence_length,
            self.reference_stats['avg_sequence_length']
        )
        shifts['sequence_length_shift'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < significance_level
        }
        
        # Check vocabulary shift
        vocab_overlap = len(
            set(current_stats.token_frequency.keys()) &
            set(self.reference_stats['token_frequency'].keys())
        ) / len(set(current_stats.token_frequency.keys()))
        shifts['vocabulary_shift'] = {
            'overlap_ratio': vocab_overlap,
            'is_significant': vocab_overlap < 0.8  # Arbitrary threshold
        }
        
        return shifts
    
    def plot_monitoring_results(self, stats: DataStats) -> None:
        """
        Create visualization of monitoring results.
        
        Args:
            stats: Current dataset statistics
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot class distribution
        pd.Series(stats.class_distribution).plot(
            kind='bar',
            ax=ax1,
            title='Class Distribution'
        )
        ax1.set_ylabel('Count')
        
        # Plot sequence length distribution
        if hasattr(self, 'sequence_lengths'):
            sns.histplot(data=self.sequence_lengths, ax=ax2)
            ax2.set_title('Sequence Length Distribution')
            ax2.set_xlabel('Length')
            ax2.set_ylabel('Count')
        
        # Plot token frequency distribution
        top_tokens = sorted(
            stats.token_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        token_df = pd.DataFrame(top_tokens, columns=['Token', 'Frequency'])
        sns.barplot(
            data=token_df,
            x='Token',
            y='Frequency',
            ax=ax3
        )
        ax3.set_title('Top Token Frequencies')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot time series if historical data available
        if self.history['num_samples']:
            dates = [datetime.fromisoformat(ts) for ts in self.history['timestamp']]
            samples = self.history['num_samples']
            ax4.plot(dates, samples, marker='o')
            ax4.set_title('Dataset Size Over Time')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'monitoring_report_{datetime.now():%Y%m%d_%H%M%S}.png')
        plt.close()
    
    def update_history(self, stats: DataStats) -> None:
        """
        Update monitoring history with new statistics.
        
        Args:
            stats: Current dataset statistics
        """
        self.history['timestamp'].append(stats.timestamp)
        self.history['num_samples'].append(stats.num_samples)
        self.history['avg_sequence_length'].append(stats.avg_sequence_length)
        self.history['num_unique_tokens'].append(stats.num_unique_tokens)
        
        # Save history to file
        history_file = self.save_dir / 'monitoring_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def generate_report(self, stats: DataStats,
                       shifts: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a detailed monitoring report.
        
        Args:
            stats: Current dataset statistics
            shifts: Distribution shift analysis results
            
        Returns:
            Formatted report string
        """
        report = ["=== Data Quality Report ===\n"]
        report.append(f"Timestamp: {stats.timestamp}\n")
        
        # Basic statistics
        report.append("\nBasic Statistics:")
        report.append(f"- Total samples: {stats.num_samples}")
        report.append("- Class distribution:")
        for cls, count in stats.class_distribution.items():
            report.append(f"  * Class {cls}: {count} ({count/stats.num_samples*100:.2f}%)")
        report.append(f"- Average sequence length: {stats.avg_sequence_length:.2f}")
        report.append(f"- Unique tokens: {stats.num_unique_tokens}")
        
        # Common patterns
        report.append("\nCommon Patterns:")
        for pattern in stats.common_patterns[:5]:
            report.append(f"- {pattern}")
            
        # Outliers
        if stats.outliers:
            report.append(f"\nOutliers detected: {len(stats.outliers)} samples")
            
        # Distribution shifts
        if shifts:
            report.append("\nDistribution Shift Analysis:")
            for metric, result in shifts.items():
                report.append(f"- {metric}:")
                for key, value in result.items():
                    report.append(f"  * {key}: {value}")
        
        report_str = '\n'.join(report)
        
        # Save report
        report_file = self.save_dir / f'report_{datetime.now():%Y%m%d_%H%M%S}.txt'
        with open(report_file, 'w') as f:
            f.write(report_str)
            
        return report_str