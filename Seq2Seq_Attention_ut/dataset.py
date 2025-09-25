import pandas as pd
import numpy as np
import spacy
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field
from collections import Counter
import random
import config

# Load spacy for tokenization
try:
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    spacy_en = spacy.load('en')

def tokenize_nl(text):
    """Tokenize natural language input"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_bash(text):
    """Simple bash tokenization - split on spaces and handle special characters"""
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

class CommandDataset(Dataset):
    def __init__(self, data, src_field, trg_field, max_len=70):
        self.data = data
        self.src_field = src_field
        self.trg_field = trg_field
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        src = example['src']
        trg = example['trg']
        cmd_type = example['type']
        
        # Convert tokens to indices
        src_indices = [self.src_field.vocab.stoi[token] for token in src]
        trg_indices = [self.trg_field.vocab.stoi[token] for token in trg]
        
        return {
            'src': torch.LongTensor(src_indices),
            'trg': torch.LongTensor(trg_indices),
            'type': cmd_type
        }

def load_json_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def create_dataset():
    # Load data from JSON files
    bash_data = load_json_data(config.BASH_DATASET_PATH)
    docker_data = load_json_data(config.DOCKER_DATASET_PATH)
    
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
    
    # Split data
    random.shuffle(processed_data)
    
    total_len = len(processed_data)
    train_len = int(total_len * config.TRAIN_SPLIT)
    valid_len = int(total_len * config.VALID_SPLIT)
    
    train_data = processed_data[:train_len]
    valid_data = processed_data[train_len:train_len + valid_len]
    test_data = processed_data[train_len + valid_len:]
    
    # Create vocabularies
    SRC = Field(
        tokenize=tokenize_nl,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=False
    )
    
    TRG = Field(
        tokenize=tokenize_bash,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=False
    )
    
    # Build vocabularies
    all_src_tokens = []
    all_trg_tokens = []
    
    for example in processed_data:
        all_src_tokens.extend(example['src'])
        all_trg_tokens.extend(example['trg'])
    
    # Create vocab from tokens
    src_counter = Counter(all_src_tokens)
    trg_counter = Counter(all_trg_tokens)
    
    # Add special tokens
    src_vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [word for word, _ in src_counter.most_common()]
    trg_vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + [word for word, _ in trg_counter.most_common()]
    
    # Create vocab mappings
    SRC.vocab = type('Vocab', (), {
        'stoi': {word: i for i, word in enumerate(src_vocab)},
        'itos': src_vocab
    })()
    
    TRG.vocab = type('Vocab', (), {
        'stoi': {word: i for i, word in enumerate(trg_vocab)},
        'itos': trg_vocab
    })()
    
    valid = data.TabularDataset(
    path=config.VALID_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])

    test = data.TabularDataset(
    path=config.TEST_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])

    SRC.build_vocab(train, valid, max_size=10000, min_freq=1)
    TRG.build_vocab(train, valid, max_size=10000, min_freq=1)
    UT.build_vocab(train, valid, max_size=300, min_freq=1)

    return train, valid, test, SRC, TRG, UT


    test = data.TabularDataset(
    path=config.TEST_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])


    
    valid = data.TabularDataset(
    path=config.VALID_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])


    SRC.build_vocab(train, 
                    vectors=torchtext.vocab.Vectors(config.GLOVE_PATH))

    TRG.build_vocab(train)

    UT.build_vocab(train)

    return train, valid, test, SRC, TRG, UT