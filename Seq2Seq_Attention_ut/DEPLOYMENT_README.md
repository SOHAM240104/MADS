# 🚀 Deployment Guide: Seq2Seq Command Generation Model

This guide explains how to deploy and run the Seq2Seq command generation model on different systems, especially with GPU acceleration.

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 2GB free disk space

### Recommended for Training
- Python 3.8+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (or Apple Silicon with MPS)
- 5GB free disk space

## 🔧 Setup Instructions

### 1. Environment Setup

```bash
# Clone or copy the project
git clone <your-repo> # or copy the Seq2Seq_Attention_ut folder

# Create virtual environment
cd Seq2Seq_Attention_ut
python -m venv .venv

# Activate environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# OR for Apple Silicon:
pip install torch torchvision torchaudio

# Install additional packages
pip install numpy scikit-learn tqdm matplotlib seaborn
```

### 2. Data Setup

**Required Files:**
```
data/
├── bash_dataset.json    # 35,477 bash command examples
└── docker_dataset.json # 2,415 docker command examples
```

**Update paths in `config.py`:**
```python
# Change these paths to match your system:
BASH_DATASET_PATH = '/path/to/your/data/bash_dataset.json'
DOCKER_DATASET_PATH = '/path/to/your/data/docker_dataset.json'
GLOVE_PATH = '/path/to/your/glove.840B.300d.txt'  # Optional
```

## 🎯 GPU Configuration

### NVIDIA CUDA Setup

1. **Install CUDA Toolkit** (11.8 or 12.1)
2. **Update config.py:**
```python
import torch

# CUDA configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
```

3. **Optimize for GPU:**
```python
# Increase batch size for better GPU utilization
BATCH_SIZE = 32  # or 64 if you have enough VRAM

# Enable mixed precision training (optional)
USE_AMP = True  # Add this to config.py
```

### Apple Silicon (MPS) Setup

The current configuration already supports MPS:
```python
# Already configured in config.py
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

### CPU-Only Setup

```python
# Force CPU usage
device = torch.device('cpu')
BATCH_SIZE = 4  # Reduce batch size for CPU
```

## ⚡ Performance Optimization

### GPU Optimization

**For NVIDIA GPUs (config.py additions):**
```python
# GPU-specific optimizations
BATCH_SIZE = 64        # Larger batch size
LEARNING_RATE = 0.003  # Higher learning rate for larger batches
NUM_WORKERS = 4        # More data loading workers

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

**Expected GPU Performance:**
- **RTX 3090/4090**: ~20-30 minutes total training
- **RTX 3080**: ~30-45 minutes total training  
- **RTX 2080 Ti**: ~45-60 minutes total training
- **Apple M1/M2**: ~90-120 minutes total training
- **CPU Only**: 4-6 hours total training

### Memory Optimization

**For Limited VRAM/RAM:**
```python
# Reduce model size
LSTM_HIDDEN_DIMENSION = 256  # Default: 512
LSTM_LAYERS = 1              # Default: 2
BATCH_SIZE = 16              # Default: 8

# Enable gradient checkpointing (add to model_new.py)
USE_GRADIENT_CHECKPOINTING = True
```

## 🏃‍♂️ Running Training

### Basic Training
```bash
python train_phased.py
```

### Advanced Training with Monitoring
```bash
# Run with logging
python train_phased.py 2>&1 | tee training.log

# Monitor GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Monitor system resources
htop  # or Activity Monitor on Mac
```

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Watch log files
tail -f logs/training_*.log

# Monitor checkpoints
ls -la checkpoints/

# Check training progress
grep "Epoch\|Loss\|EM Score" logs/training_*.log
```

### Key Metrics to Monitor
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease, watch for overfitting
- **Exact Match (EM) Score**: Target >70% for good performance
- **BLEU Score**: Target >0.8 for good performance

## 🔧 Configuration for Different Systems

### High-Performance GPU System
```python
# config.py optimizations
BATCH_SIZE = 128
LEARNING_RATE = 0.005
N_EPOCHS = 15
LSTM_HIDDEN_DIMENSION = 768
LSTM_LAYERS = 3
```

### Medium GPU System
```python
# config.py for RTX 3060/similar
BATCH_SIZE = 32
LEARNING_RATE = 0.002
N_EPOCHS = 12
LSTM_HIDDEN_DIMENSION = 512
LSTM_LAYERS = 2
```

### Limited Resources System
```python
# config.py for CPU or low VRAM
BATCH_SIZE = 8
LEARNING_RATE = 0.001
N_EPOCHS = 8
LSTM_HIDDEN_DIMENSION = 256
LSTM_LAYERS = 1
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size
BATCH_SIZE = 16  # or lower

# Enable gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

**Slow Training:**
```python
# Increase batch size (if memory allows)
BATCH_SIZE = 64

# Use more workers
NUM_WORKERS = 4

# Enable mixed precision
USE_AMP = True
```

**Poor Performance:**
```python
# Increase model capacity
LSTM_HIDDEN_DIMENSION = 768
LSTM_LAYERS = 3

# Train longer
N_EPOCHS = 20

# Lower learning rate
LEARNING_RATE = 0.0005
```

## 📁 File Structure After Deployment

```
Seq2Seq_Attention_ut/
├── config.py              # ✏️ MODIFY PATHS HERE
├── dataset_new.py          # Data loading
├── model_new.py           # Model architecture  
├── train_phased.py        # 🚀 RUN THIS
├── inference.py           # For predictions
├── evaluation.py          # Evaluation metrics
├── advanced_features.py   # Advanced features
├── checkpoints/           # 📁 Model checkpoints saved here
├── logs/                  # 📁 Training logs
├── .venv/                # Python environment
└── README.md             # Documentation

data/                      # ✏️ UPDATE THESE PATHS
├── bash_dataset.json
└── docker_dataset.json
```

## 🎯 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Create virtual environment
- [ ] Install PyTorch (with CUDA if available)
- [ ] Install additional packages
- [ ] Update data paths in `config.py`
- [ ] Adjust batch size for your system
- [ ] Run `python train_phased.py`
- [ ] Monitor training progress
- [ ] Use `inference.py` for predictions

## 🚀 Expected Results

**After successful training:**
- **Bash Commands**: ~75-80% Exact Match
- **Docker Commands**: ~65-70% Exact Match  
- **Overall BLEU Score**: ~0.82-0.87
- **Model Size**: ~80MB checkpoint file

## 💡 Tips for Best Results

1. **Use GPU** if available (10-20x faster)
2. **Increase batch size** on powerful systems
3. **Monitor for overfitting** (validation loss increases)
4. **Save intermediate checkpoints** for recovery
5. **Validate results** on held-out test data

---

**Need help?** Check the training logs in `logs/` directory or the detailed README.md in the project root.