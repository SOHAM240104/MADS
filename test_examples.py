#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Test script for Seq2Seq model with custom examples
"""

import sys
import os
import torch

# Add the necessary paths
sys.path.append('./Seq2Seq')
sys.path.append('./clai/utils')

# Import necessary modules
import config
import dataset
import model

# Example natural language queries to test
TEST_EXAMPLES = [
 
    "show the current date and time"
    "list all the running process"
]

def main():
    # Fix the config paths to use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config.TRAIN_PATH = os.path.join(base_dir, 'data', 'train.csv')
    config.VALID_PATH = os.path.join(base_dir, 'data', 'valid.csv')
    config.TEST_PATH = os.path.join(base_dir, 'data', 'test.csv')
    
    # Create dataset to get vocabulary
    _, _, _, SRC, TRG = dataset.create_dataset()
    
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
        
    model_rnn.load_state_dict(torch.load(model_path))
    model_rnn.eval()
    
    # Helper function to translate a sentence
    def translate_sentence(sentence, max_len=50):
        tokens = dataset.tokenize_nl(sentence.lower())
        # Add <sos> and <eos>
        tokens = ['<sos>'] + tokens + ['<eos>']

        src_indexes = [SRC.vocab.stoi.get(tok, SRC.vocab.stoi[SRC.unk_token]) for tok in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(config.device)  # [src_len, 1]

        with torch.no_grad():
            hidden, cell = model_rnn.encoder(src_tensor)

        trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(config.device)
            with torch.no_grad():
                output, hidden, cell = model_rnn.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == TRG.vocab.stoi[TRG.eos_token]:
                break

        trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
        # remove <sos>
        return trg_tokens[1:]
    
    # Test each example
    print("\nTesting Seq2Seq model with examples:\n")
    print("-" * 80)
    
    for i, example in enumerate(TEST_EXAMPLES):
        pred_tokens = translate_sentence(example)
        # Strip special tokens and <unk>
        cleaned = [t for t in pred_tokens if t not in {TRG.init_token, TRG.eos_token, TRG.pad_token, '<unk>'}]
        
        print(f"Example {i+1}:")
        print(f"Input : {example}")
        print(f"Output: {' '.join(cleaned)}")
        print("-" * 80)

if __name__ == "__main__":
    main()