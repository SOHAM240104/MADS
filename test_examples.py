#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Test script for Seq2Seq model with custom examples
"""

import sys
import os
import torch
import spacy

# Add the necessary paths first
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'Seq2Seq'))
sys.path.append(os.path.join(base_dir, 'clai/utils'))

# Import necessary modules
from Seq2Seq import config
from Seq2Seq import dataset
from Seq2Seq import model

# Initialize the spacy model
try:
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading Spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load('en_core_web_sm')

def tokenize_nl(text):
    # Normalize text
    text = text.lower().strip()
    # Tokenize with spacy
    tokens = [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_space]
    # Reverse the tokens as per the training data
    tokens = tokens[::-1]
    return tokens

# Example natural language queries to test
TEST_EXAMPLES = [
    "find files in current directory that end with .txt",
    "show current date and time",
    "list all files in home directory"
]

def main():
    # Fix the config paths to use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config.TRAIN_PATH = os.path.join(base_dir, 'data', 'train.csv')
    config.VALID_PATH = os.path.join(base_dir, 'data', 'valid.csv')
    config.TEST_PATH = os.path.join(base_dir, 'data', 'test.csv')
    
    # Create dataset to get vocabulary
    _, _, _, SRC, TRG = dataset.create_dataset()
    
    # Debug vocabulary
    print("\nVocabulary Debug Info:")
    print("Source vocab size:", len(SRC.vocab))
    print("Target vocab size:", len(TRG.vocab))
    print("\nSample source vocabulary:")
    print(list(SRC.vocab.stoi.items())[:10])
    print("\nSample target vocabulary:")
    print(list(TRG.vocab.stoi.items())[:10])
    
    # Set up model architecture
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = config.ENCODER_EMBEDDING_DIMENSION
    DEC_EMB_DIM = config.DECODER_EMBEDDING_DIMENSION
    HID_DIM = config.LSTM_HIDDEN_DIMENSION
    N_LAYERS = config.LSTM_LAYERS
    ENC_DROPOUT = config.ENCODER_DROPOUT
    DEC_DROPOUT = config.DECODER_DROPOUT

    # Initialize model components
    enc = model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model_rnn = model.Seq2seq(enc, dec, config.device).to(config.device)
    
    # Load trained model
    model_path = os.path.join(base_dir, 'Seq2Seq', 'model_seq2seq_1.bin')
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return

    # Load the state dict
    state_dict = torch.load(model_path, map_location=config.device)
    
    # Helper function to resize embedding and linear layers
    def resize_parameter(old_param, new_size, dimension=0):
        if len(old_param.shape) == 2:
            # For embeddings and linear layers, we want to preserve the learned weights
            new_param = torch.zeros(new_size, old_param.shape[1], device=old_param.device)
            new_param[:old_param.shape[0], :] = old_param
            return new_param
        else:
            # For bias vectors
            new_param = torch.zeros(new_size, device=old_param.device)
            new_param[:old_param.shape[0]] = old_param
            return new_param

    # Adjust parameters for vocabulary size differences
    for key in list(state_dict.keys()):
        if 'decoder.embedding.weight' in key:
            old_size = state_dict[key].shape[0]
            if old_size != OUTPUT_DIM:
                state_dict[key] = resize_parameter(state_dict[key], OUTPUT_DIM)
        elif 'decoder.out.weight' in key:
            old_size = state_dict[key].shape[0]
            if old_size != OUTPUT_DIM:
                state_dict[key] = resize_parameter(state_dict[key], OUTPUT_DIM)
        elif 'decoder.out.bias' in key:
            old_size = state_dict[key].shape[0]
            if old_size != OUTPUT_DIM:
                state_dict[key] = resize_parameter(state_dict[key], OUTPUT_DIM)

    # Load the modified state dict
    model_rnn.load_state_dict(state_dict)
    model_rnn.eval()
    
    # Helper function to translate a sentence
    def translate_sentence(sentence, max_len=50):
        # Normalize and tokenize input
        tokens = tokenize_nl(sentence)
        # Add <sos> and <eos>
        tokens = ['<sos>'] + tokens + ['<eos>']

        # Convert to indexes
        src_indexes = [SRC.vocab.stoi.get(tok, SRC.vocab.stoi[SRC.unk_token]) for tok in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(config.device)

        with torch.no_grad():
            # Get encoder outputs
            hidden, cell = model_rnn.encoder(src_tensor)
            
            # Initialize decoder input
            trg_tensor = torch.LongTensor([TRG.vocab.stoi[TRG.init_token]]).to(config.device)
            trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

            # Generate tokens one at a time
            for _ in range(max_len):
                # Get decoder output
                output, hidden, cell = model_rnn.decoder(trg_tensor, hidden, cell)
                
                # Get probabilities and top prediction
                output_dist = torch.nn.functional.softmax(output, dim=1)
                top_prob, top_idx = output_dist.data.topk(1)
                top_idx = top_idx.item()
                
                # Add prediction to outputs
                trg_indexes.append(top_idx)
                trg_tensor = torch.LongTensor([top_idx]).to(config.device)
                
                # Break if end token
                if top_idx == TRG.vocab.stoi[TRG.eos_token]:
                    break
            
        # Convert indexes back to tokens, excluding <sos> but keeping <eos>
        trg_tokens = []
        for idx in trg_indexes[1:]:  # Skip initial <sos>
            if idx < len(TRG.vocab.itos):
                token = TRG.vocab.itos[idx]
                if token not in {'<sos>', '<pad>'}:  # Keep <eos> but remove other special tokens
                    trg_tokens.append(token)
            
            if token == '<eos>':
                break
                
        return trg_tokens

        trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
        # remove <sos>
        return trg_tokens[1:]
    
    # Test each example
    print("\nTesting Seq2Seq model with examples:\n")
    print("-" * 80)
    
    for i, example in enumerate(TEST_EXAMPLES):
        # Tokenize input
        src_tokens = tokenize_nl(example.lower())
        print(f"\nExample {i+1}:")
        print(f"Input : {example}")
        print(f"Tokenized Input: {src_tokens}")
        
        # Get model prediction
        pred_tokens = translate_sentence(example)
        
        # Get token IDs
        token_ids = [TRG.vocab.stoi.get(token, TRG.vocab.stoi[TRG.unk_token]) for token in pred_tokens]
        
        # Strip special tokens and <unk>
        cleaned = [t for t in pred_tokens if t not in {TRG.init_token, TRG.eos_token, TRG.pad_token, '<unk>'}]
        
        print(f"Raw Output: {pred_tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Cleaned Output: {' '.join(cleaned)}")
        print("-" * 80)

if __name__ == "__main__":
    main()