# Advanced Sequence-to-Sequence Model for Command Generation

This project implements a state-of-the-art sequence-to-sequence model with bi-directional LSTM encoder and attention mechanism for generating bash and docker commands from natural language descriptions.

## 🏗️ Architecture

### Model Components

1. **Bi-directional LSTM Encoder**
   - Multi-layer bi-directional LSTM for capturing context from both directions
   - Supports variable-length sequences with proper padding
   - Projects combined forward/backward states to decoder dimensions

2. **Bahdanau Attention Mechanism**
   - Additive attention for better alignment between input and output
   - Computed attention weights for interpretability
   - Context vector computation for each decoding step

3. **LSTM Decoder**
   - Unidirectional LSTM with attention integration
   - Combines embedding, context, and hidden states for output prediction
   - Teacher forcing during training, beam search during inference

### Professional AI Engineering Features

- **Phased Training**: Train on bash-only data first, then fine-tune on combined dataset
- **Beam Search Decoding**: Improved inference quality with configurable beam size
- **Advanced Optimization**: Learning rate scheduling, gradient clipping, early stopping
- **Model Checkpointing**: Save best models based on multiple criteria
- **Comprehensive Evaluation**: Exact Match, BLEU scores, per-class metrics
- **Label Smoothing**: Improved generalization during training
- **Data Augmentation**: Synonym replacement, insertion, deletion techniques

## 📁 Project Structure

```
Seq2Seq_Attention_ut/
├── config.py              # Configuration parameters
├── dataset_new.py          # Dataset loading and preprocessing
├── model_new.py           # Model architecture implementation
├── train_phased.py        # Phased training script
├── evaluation.py          # Comprehensive evaluation metrics
├── advanced_features.py   # Professional AI enhancements
├── inference.py           # Inference and prediction script
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchtext numpy scikit-learn spacy tqdm matplotlib

# Install spacy model
python -m spacy download en_core_web_sm
```

### 2. Data Preparation

Ensure your JSON datasets are in the correct location:
```
../data/bash_dataset.json
../data/docker_dataset.json
```

### 3. Training

```bash
# Run phased training
python train_phased.py

# Monitor training progress
tail -f logs/training_*.log
```

### 4. Inference

```bash
# Interactive mode
python inference.py --model_path checkpoints/phase_2_best_model.pt --interactive

# Single prediction
python inference.py --model_path checkpoints/phase_2_best_model.pt --input "List all Python files in the directory"

# Batch prediction
python inference.py --model_path checkpoints/phase_2_best_model.pt --input_file queries.txt --output_file results.json
```

## 📊 Model Configuration

Key hyperparameters in `config.py`:

```python
# Model Architecture
LSTM_HIDDEN_DIMENSION = 512
LSTM_LAYERS = 2
ENCODER_EMBEDDING_DIMENSION = 300
DECODER_EMBEDDING_DIMENSION = 300

# Training
BATCH_SIZE = 8
LEARNING_RATE = 0.001
N_EPOCHS = 10
CLIP = 1

# Data Splits
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
```

## 🔬 Training Strategy

### Phase 1: Bash-Only Training
- Train exclusively on bash command dataset (35,477 examples)
- Establish strong baseline understanding of command syntax
- Learn fundamental patterns in command generation

### Phase 2: Combined Fine-tuning
- Fine-tune on combined bash + docker dataset (37,892 examples)
- Reduced learning rate for stable convergence
- Transfer learning from Phase 1 model

### Advanced Training Features
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Save best models based on validation metrics

## 📈 Evaluation Metrics

### Overall Metrics
- **Exact Match (EM)**: Percentage of predictions that exactly match targets
- **BLEU Score**: Measures n-gram overlap between predictions and targets
- **Loss**: Cross-entropy loss on validation set

### Per-Class Analysis
- **Bash Commands**: Precision, Recall, F1-score
- **Docker Commands**: Precision, Recall, F1-score
- **Classification Report**: Detailed per-class metrics

### Sample Output
```
EVALUATION RESULTS
================================================================================

OVERALL METRICS:
  Total Examples: 3789
  Average Loss: 2.3456
  Exact Match Score: 0.7234 (72.34%)
  BLEU Score: 0.8123

PER-TYPE METRICS:
  BASH Commands (3254 examples):
    Exact Match: 0.7456 (74.56%)
    BLEU Score: 0.8234

  DOCKER Commands (535 examples):
    Exact Match: 0.6123 (61.23%)
    BLEU Score: 0.7456
```

## 🔧 Advanced Features

### Beam Search Inference
```python
# Configure beam search parameters
beam_decoder = BeamSearchDecoder(
    model=model,
    beam_size=5,
    max_length=70,
    eos_token=vocab.word2idx['<eos>'],
    sos_token=vocab.word2idx['<sos>']
)
```

### Model Ensemble
```python
# Combine multiple models for improved performance
ensemble = ModelEnsemble(
    models=[model1, model2, model3],
    weights=[0.5, 0.3, 0.2]
)
```

### Data Augmentation
```python
# Apply augmentation techniques
augmented_tokens = DataAugmentation.synonym_replacement(
    tokens, vocab, replacement_prob=0.1
)
```

## 🎯 Professional Recommendations

### As a Pro AI Engineer, here are additional suggestions:

1. **Model Improvements**
   - Implement Transformer architecture for better long-range dependencies
   - Add copy mechanism for handling out-of-vocabulary tokens
   - Experiment with pre-trained language models (BERT, GPT)

2. **Data Enhancements**
   - Implement active learning for efficient data annotation
   - Add data validation and quality checks
   - Create synthetic data using paraphrasing models

3. **Production Readiness**
   - Implement model versioning and A/B testing
   - Add comprehensive monitoring and alerting
   - Create API endpoints with proper authentication

4. **Evaluation Improvements**
   - Implement semantic similarity metrics
   - Add execution-based evaluation (if commands can be safely run)
   - Create domain-specific evaluation metrics

5. **Scalability**
   - Implement distributed training for larger datasets
   - Add model quantization for inference speed
   - Create efficient batching strategies

## 📋 Usage Examples

### Training
```bash
# Full training with default settings
python train_phased.py

# View training progress
tensorboard --logdir logs/
```

### Inference Examples
```python
# Load model for inference
inference = CommandInference('checkpoints/phase_2_best_model.pt')

# Generate commands
result = inference.predict("Find all Python files larger than 1MB")
print(f"Command: {result['prediction']}")
# Output: find . -name "*.py" -size +1M
```

## 📚 Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn
- spaCy
- tqdm
- matplotlib (for visualization)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with proper tests
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Monitoring and Debugging

### Training Logs
- Comprehensive logging with timestamps
- Progress tracking for each epoch
- Metric logging for analysis

### Model Debugging
- Attention weight visualization
- Gradient norm monitoring
- Layer-wise analysis tools

### Performance Monitoring
- Memory usage tracking
- Training speed benchmarks
- Model size optimization

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config.py
   - Use gradient accumulation
   - Enable mixed precision training

2. **Poor Performance on Docker Commands**
   - Increase docker data augmentation
   - Adjust class weights in loss function
   - Use focal loss for class imbalance

3. **Slow Training**
   - Enable distributed training
   - Use larger batch sizes with learning rate scaling
   - Implement gradient checkpointing

### Performance Tips

1. **Optimize Data Loading**
   - Use multiple workers in DataLoader
   - Implement efficient preprocessing
   - Cache preprocessed data

2. **Model Optimization**
   - Use mixed precision training
   - Implement gradient accumulation
   - Enable torch.jit.script for inference

3. **Memory Management**
   - Use gradient checkpointing
   - Clear cache regularly
   - Monitor memory usage

---

**Built with ❤️ for advancing natural language to command translation**