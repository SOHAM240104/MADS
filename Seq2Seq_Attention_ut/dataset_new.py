import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import random
import config

# Load spacy for tokenization
try:
    import spacy
    spacy_en = spacy.load('en_core_web_sm')
except (OSError, ImportError):
    try:
        import spacy
        spacy_en = spacy.load('en')
    except (OSError, ImportError):
        print("Warning: Spacy not available. Using basic tokenization.")
        spacy_en = None

def tokenize_nl(text):
    """Tokenize natural language input"""
    # Handle non-string inputs
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    if spacy_en:
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    else:
        # Fallback tokenization
        return text.lower().split()

def tokenize_bash(text):
    """Simple bash tokenization - split on spaces and handle special characters"""
    # Handle non-string inputs
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Basic tokenization for bash commands
    tokens = []
    current_token = ""
    in_quotes = False
    
    for char in text:
        if char in ['"', "'"]:
            in_quotes = not in_quotes
            current_token += char
        elif char == ' ' and not in_quotes:
            if current_token:
                tokens.append(current_token.lower())
                current_token = ""
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token.lower())
    
    return tokens

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word_count = {'<pad>': 1, '<unk>': 1, '<sos>': 1, '<eos>': 1}
        self.n_words = 4  # Count default tokens

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word_count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1

class CommandDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, max_len=70):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        src = example['src']
        trg = example['trg']
        cmd_type = example['type']
        
        # Convert tokens to indices
        src_indices = [self.src_vocab.word2idx.get(token, self.src_vocab.word2idx['<unk>']) for token in src]
        trg_indices = [self.trg_vocab.word2idx.get(token, self.trg_vocab.word2idx['<unk>']) for token in trg]
        
        # Truncate if too long
        src_indices = src_indices[:self.max_len]
        trg_indices = trg_indices[:self.max_len]
        
        return {
            'src': torch.LongTensor(src_indices),
            'trg': torch.LongTensor(trg_indices),
            'type': cmd_type,
            'src_len': len(src_indices),
            'trg_len': len(trg_indices)
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]
    types = [item['type'] for item in batch]
    src_lengths = [item['src_len'] for item in batch]
    trg_lengths = [item['trg_len'] for item in batch]
    
    # Pad sequences
    src_batch = pad_sequence(src_batch, batch_first=False, padding_value=0)
    trg_batch = pad_sequence(trg_batch, batch_first=False, padding_value=0)
    
    return {
        'src': src_batch,
        'trg': trg_batch,
        'types': types,
        'src_lengths': src_lengths,
        'trg_lengths': trg_lengths
    }

def load_json_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def create_dataset_and_vocabs():
    """Create datasets and vocabularies from JSON files"""
    print("Loading data from JSON files...")
    
    # Load data from JSON files
    bash_data = load_json_data(config.BASH_DATASET_PATH)
    docker_data = load_json_data(config.DOCKER_DATASET_PATH)
    
    print(f"Loaded {len(bash_data)} bash examples and {len(docker_data)} docker examples")
    
    # Process data
    processed_data = []
    
    # Process bash data
    for example in bash_data:
        src_tokens = tokenize_nl(example['input'])
        trg_tokens = ['<sos>'] + tokenize_bash(example['output']) + ['<eos>']
        processed_data.append({
            'src': src_tokens,
            'trg': trg_tokens,
            'type': 'bash'
        })
    
    # Process docker data  
    for example in docker_data:
        src_tokens = tokenize_nl(example['input'])
        trg_tokens = ['<sos>'] + tokenize_bash(example['output']) + ['<eos>']
        processed_data.append({
            'src': src_tokens,
            'trg': trg_tokens,
            'type': 'docker'
        })
    
    print(f"Total processed examples: {len(processed_data)}")
    
    # Create vocabularies
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    # Build vocabularies from all data
    for example in processed_data:
        src_vocab.add_sentence(example['src'])
        trg_vocab.add_sentence(example['trg'])
    
    print(f"Source vocabulary size: {src_vocab.n_words}")
    print(f"Target vocabulary size: {trg_vocab.n_words}")
    
    # Split data
    random.seed(42)  # For reproducibility
    random.shuffle(processed_data)
    
    total_len = len(processed_data)
    train_len = int(total_len * config.TRAIN_SPLIT)
    valid_len = int(total_len * config.VALID_SPLIT)
    
    train_data = processed_data[:train_len]
    valid_data = processed_data[train_len:train_len + valid_len]
    test_data = processed_data[train_len + valid_len:]
    
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    return train_data, valid_data, test_data, src_vocab, trg_vocab

def create_bash_only_dataset():
    """Create dataset with only bash commands for Phase 1 training"""
    print("Loading bash-only data for Phase 1...")
    
    bash_data = load_json_data(config.BASH_DATASET_PATH)
    
    processed_data = []
    for example in bash_data:
        src_tokens = tokenize_nl(example['input'])
        trg_tokens = ['<sos>'] + tokenize_bash(example['output']) + ['<eos>']
        processed_data.append({
            'src': src_tokens,
            'trg': trg_tokens,
            'type': 'bash'
        })
    
    # Create vocabularies
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    for example in processed_data:
        src_vocab.add_sentence(example['src'])
        trg_vocab.add_sentence(example['trg'])
    
    # Split data
    random.seed(42)
    random.shuffle(processed_data)
    
    total_len = len(processed_data)
    train_len = int(total_len * config.TRAIN_SPLIT)
    valid_len = int(total_len * config.VALID_SPLIT)
    
    train_data = processed_data[:train_len]
    valid_data = processed_data[train_len:train_len + valid_len]
    test_data = processed_data[train_len + valid_len:]
    
    print(f"Bash-only - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    return train_data, valid_data, test_data, src_vocab, trg_vocab

def get_data_loaders(train_data, valid_data, test_data, src_vocab, trg_vocab, batch_size=None):
    """Create DataLoaders for training"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_dataset = CommandDataset(train_data, src_vocab, trg_vocab, config.MAX_LEN)
    valid_dataset = CommandDataset(valid_data, src_vocab, trg_vocab, config.MAX_LEN)
    test_dataset = CommandDataset(test_data, src_vocab, trg_vocab, config.MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)  # Set to 0 for compatibility
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    return train_loader, valid_loader, test_loader