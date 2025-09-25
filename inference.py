#!/usr/bin/env python3
"""
Inference script for generating bash/docker commands from natural language queries
Works with the Seq2Seq_Attention_ut model
"""

import sys
import os
sys.path.append('Seq2Seq_Attention_ut')

import torch
import pickle
from torchtext.data import Field
import spacy
import random
import numpy as np

# Import model components
from Seq2Seq_Attention_ut import config, model, dataset

class CommandGenerator:
    def __init__(self, model_path=None, vocab_path=None):
        """Initialize the command generator with trained model and vocabularies"""
        self.device = config.device
        self.model = None
        self.src_vocab = None
        self.trg_vocab = None
        self.utilities_dict = None
        
        # Set random seeds for reproducibility
        self.set_seeds(1234)
        
        # Try to use available trained models
        if model_path is None:
            model_path = self.find_best_model()
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {model_path}")
        
        self.load_model_and_vocab(model_path)
    
    def set_seeds(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def find_best_model(self):
        """Find the best available trained model"""
        possible_models = [
            'best_model_seq2seq_attention.bin',
            'model_seq2seq_attention.bin',
            'Seq2Seq/model_seq2seq_1.bin'
        ]
        
        for model_file in possible_models:
            if os.path.exists(model_file):
                return model_file
        
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    def load_model_and_vocab(self, model_path):
        """Load the trained model and vocabularies"""
        try:
            # Create dataset to get vocabularies
            print("Loading vocabularies...")
            train, valid, test, SRC, TRG, UT = dataset.create_dataset()
            
            self.src_vocab = SRC
            self.trg_vocab = TRG
            
            # Try to load utilities
            try:
                utilities = pickle.load(open('list_of_utilities.pkl', 'rb'))
                self.utilities_dict = {}
                for i in range(len(utilities)):
                    self.utilities_dict[i] = TRG.vocab.stoi[utilities[i]]
            except FileNotFoundError:
                print("Warning: utilities file not found. Creating empty utilities dict.")
                self.utilities_dict = {}
            
            # Initialize model architecture
            INPUT_DIM = len(SRC.vocab)
            OUTPUT_DIM = len(TRG.vocab)
            ENC_EMB_DIM = config.ENCODER_EMBEDDING_DIMENSION
            DEC_EMB_DIM = config.DECODER_EMBEDDING_DIMENSION
            HID_DIM = config.LSTM_HIDDEN_DIMENSION
            ENC_DROPOUT = config.ENCODER_DROPOUT
            DEC_DROPOUT = config.DECODER_DROPOUT
            
            # Create model components
            attn = model.Attention(HID_DIM, HID_DIM)
            enc = model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, 
                              ENC_DROPOUT, len(self.utilities_dict), self.utilities_dict)
            dec = model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, attn)
            
            self.model = model.Seq2Seq(enc, dec, self.device).to(self.device)
            
            # Load trained weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_query(self, query):
        """Preprocess natural language query for the model"""
        # Tokenize using the same preprocessing as training
        tokenized = self.src_vocab.preprocess(query)
        
        # Convert tokens to indices
        indexed = [self.src_vocab.vocab.stoi[token] for token in tokenized]
        
        # Convert to tensor and add batch dimension
        tensor = torch.LongTensor(indexed).unsqueeze(1).to(self.device)
        
        return tensor
    
    def generate_command(self, query, max_length=50):
        """Generate a command from natural language query"""
        try:
            # Preprocess the query
            src_tensor = self.preprocess_query(query)
            
            with torch.no_grad():
                # Get encoder outputs
                encoder_outputs, hidden, uv, ut = self.model.encoder(src_tensor)
                
                # Initialize decoder input with SOS token
                trg_indexes = [self.trg_vocab.vocab.stoi[self.trg_vocab.init_token]]
                
                # Generate tokens one by one
                for i in range(max_length):
                    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                    
                    with torch.no_grad():
                        output, hidden = self.model.decoder(trg_tensor, hidden, encoder_outputs, uv)
                    
                    pred_token = output.argmax(1).item()
                    trg_indexes.append(pred_token)
                    
                    # Stop if EOS token is generated
                    if pred_token == self.trg_vocab.vocab.stoi[self.trg_vocab.eos_token]:
                        break
                
                # Convert indices back to tokens
                trg_tokens = [self.trg_vocab.vocab.itos[i] for i in trg_indexes]
                
                # Remove special tokens and join
                command = ' '.join(trg_tokens[1:-1])  # Remove SOS and EOS tokens
                
                return command.strip()
                
        except Exception as e:
            print(f"Error generating command: {e}")
            return "Error: Could not generate command"
    
    def interactive_mode(self):
        """Interactive mode for testing queries"""
        print("\\n" + "="*60)
        print("COMMAND GENERATOR - Interactive Mode")
        print("="*60)
        print("Enter natural language queries to generate bash/docker commands")
        print("Type 'quit' or 'exit' to stop")
        print("="*60 + "\\n")
        
        while True:
            try:
                query = input("\\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print(f"\\nQuery: {query}")
                print("Generating command...")
                
                command = self.generate_command(query)
                print(f"Generated Command: {command}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\\n\\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function to run the command generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate bash/docker commands from natural language")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to trained model file")
    parser.add_argument("--query", type=str, default=None,
                       help="Single query to process (non-interactive mode)")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the generator
        generator = CommandGenerator(model_path=args.model)
        
        if args.query:
            # Single query mode
            print(f"Query: {args.query}")
            command = generator.generate_command(args.query)
            print(f"Generated Command: {command}")
        else:
            # Interactive mode
            generator.interactive_mode()
            
    except Exception as e:
        print(f"Error initializing command generator: {e}")
        print("\\nPlease ensure:")
        print("1. You have trained models available")
        print("2. The dataset files are accessible")
        print("3. All dependencies are installed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())