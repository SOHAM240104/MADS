"""
Utility functions for using the trained command classifier
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Union

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from simple_classifier import CommandClassifier, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommandClassifierPredictor:
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.config = Config()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Load model
        self.model = CommandClassifier(self.config.model_name)
        checkpoint = torch.load(
            self.model_dir / 'best_model.pt',
            map_location=self.config.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        self.model.eval()
        
        # Load metrics
        with open(self.model_dir / 'metrics.json') as f:
            self.metrics = json.load(f)
            logger.info(f"Model metrics: {self.metrics}")
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict whether a command is bash or docker."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        # Map prediction to label
        label = 'docker' if prediction == 1 else 'bash'
        
        return {
            'command_type': label,
            'confidence': confidence,
            'text': text
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Predict command types for a batch of texts."""
        return [self.predict(text) for text in texts]
    
    def evaluate(self, test_texts: List[str], test_labels: List[int],
                plot_confusion: bool = True) -> Dict[str, float]:
        """Evaluate model on test data."""
        predictions = []
        confidences = []
        
        # Get predictions
        for text in test_texts:
            result = self.predict(text)
            pred = 1 if result['command_type'] == 'docker' else 0
            predictions.append(pred)
            confidences.append(result['confidence'])
        
        # Calculate metrics
        report = classification_report(
            test_labels, predictions,
            target_names=['bash', 'docker'],
            output_dict=True
        )
        
        if plot_confusion:
            # Plot confusion matrix
            cm = confusion_matrix(test_labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                xticklabels=['bash', 'docker'],
                yticklabels=['bash', 'docker']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.model_dir / 'confusion_matrix.png')
            plt.close()
        
        # Calculate additional metrics
        avg_confidence = np.mean(confidences)
        metrics = {
            'accuracy': report['accuracy'],
            'bash_f1': report['bash']['f1-score'],
            'docker_f1': report['docker']['f1-score'],
            'avg_confidence': avg_confidence
        }
        
        return metrics

def example_usage():
    """Example of how to use the classifier."""
    # Initialize predictor
    predictor = CommandClassifierPredictor('models')
    
    # Single prediction
    command = "List all running containers in docker"
    result = predictor.predict(command)
    print(f"\nCommand: {command}")
    print(f"Prediction: {result['command_type']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Batch prediction
    commands = [
        "Show all files in current directory",
        "Build a docker image from Dockerfile",
        "Create a new directory",
        "Pull latest nginx image"
    ]
    results = predictor.predict_batch(commands)
    
    print("\nBatch predictions:")
    for cmd, res in zip(commands, results):
        print(f"{res['command_type']} ({res['confidence']:.2%}): {cmd}")

if __name__ == "__main__":
    example_usage()