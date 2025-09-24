"""
Configuration for Seq2Seq Transformer model
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer

class TransformerConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 512  # Transformer hidden dimension
        self.n_heads = 8    # Number of attention heads
        self.n_layers = 6   # Number of transformer layers
        self.d_ff = 2048    # Feed-forward dimension
        self.dropout = 0.1  # Dropout rate
        
        # Tokenizer and vocab
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token_id = self.tokenizer.cls_token_id  # Use [CLS] as start token
        self.eos_token_id = self.tokenizer.sep_token_id  # Use [SEP] as end token
        
        # Training parameters
        self.batch_size = 32
        self.max_length = 128  # Maximum sequence length
        self.learning_rate = 1e-4
        self.n_epochs = 20
        self.warmup_steps = 4000
        self.label_smoothing = 0.1
        self.clip_grad = 1.0  # Maximum gradient norm
        
        # Paths
        self.data_dir = Path("/Users/mohamedaamir/Documents/MADS/data")
        self.bash_data = self.data_dir / "bash_dataset.json"
        self.docker_data = self.data_dir / "docker_dataset.json"
        self.model_dir = Path("checkpoints")
        self.model_dir.mkdir(exist_ok=True)
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        
        # Device configuration
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')
        
        # Data sampling
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # Save paths
        self.model_path = self.model_dir / "transformer_model.pt"
        self.tokenizer_path = self.model_dir / "tokenizer"
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

config = TransformerConfig()
