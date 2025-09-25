import torch
import json
import argparse
from pathlib import Path

import config
from dataset_new import tokenize_nl, tokenize_bash, Vocabulary
from model_new import Seq2SeqWithAttention
from advanced_features import BeamSearchDecoder

class CommandInference:
    """
    Inference class for the trained sequence-to-sequence model
    """
    def __init__(self, model_path, vocab_info_path=None, device=None):
        self.device = device or config.device
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract vocabulary info
        if 'vocab_info' in checkpoint:
            vocab_info = checkpoint['vocab_info']
        elif vocab_info_path:
            with open(vocab_info_path, 'r') as f:
                vocab_info = json.load(f)
        else:
            raise ValueError("Vocabulary information not found in checkpoint or separate file")
        
        # Recreate vocabularies
        self.src_vocab = self._create_vocab_from_info(vocab_info.get('src_word2idx', {}))
        self.trg_vocab = self._create_vocab_from_info(vocab_info['trg_word2idx'])
        
        # Create model
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            self.model = Seq2SeqWithAttention(
                src_vocab_size=model_config['src_vocab_size'],
                trg_vocab_size=model_config['trg_vocab_size'],
                encoder_emb_dim=model_config['encoder_emb_dim'],
                decoder_emb_dim=model_config['decoder_emb_dim'],
                encoder_hidden_dim=model_config['encoder_hidden_dim'],
                decoder_hidden_dim=model_config['decoder_hidden_dim'],
                encoder_num_layers=model_config['encoder_num_layers'],
                attention_dim=model_config['attention_dim'],
                encoder_dropout=0.0,  # No dropout during inference
                decoder_dropout=0.0,
                device=self.device
            ).to(self.device)
        else:
            # Fallback to config values
            self.model = Seq2SeqWithAttention(
                src_vocab_size=len(vocab_info.get('src_word2idx', self.src_vocab.word2idx)),
                trg_vocab_size=len(vocab_info['trg_word2idx']),
                encoder_emb_dim=config.ENCODER_EMBEDDING_DIMENSION,
                decoder_emb_dim=config.DECODER_EMBEDDING_DIMENSION,
                encoder_hidden_dim=config.LSTM_HIDDEN_DIMENSION,
                decoder_hidden_dim=config.LSTM_HIDDEN_DIMENSION,
                encoder_num_layers=config.LSTM_LAYERS,
                attention_dim=config.LSTM_HIDDEN_DIMENSION,
                encoder_dropout=0.0,
                decoder_dropout=0.0,
                device=self.device
            ).to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Initialize beam search decoder
        self.beam_decoder = BeamSearchDecoder(
            model=self.model,
            beam_size=5,
            max_length=config.MAX_LEN,
            eos_token=self.trg_vocab.word2idx['<eos>'],
            sos_token=self.trg_vocab.word2idx['<sos>']
        )
        
        print(f"Model loaded successfully!")
        print(f"Source vocabulary size: {self.src_vocab.n_words}")
        print(f"Target vocabulary size: {self.trg_vocab.n_words}")
    
    def _create_vocab_from_info(self, word2idx):
        """Create vocabulary object from word2idx mapping"""
        vocab = Vocabulary()
        vocab.word2idx = word2idx
        vocab.idx2word = {idx: word for word, idx in word2idx.items()}
        vocab.n_words = len(word2idx)
        return vocab
    
    def preprocess_input(self, natural_language_query):
        """
        Preprocess natural language input
        Args:
            natural_language_query: str - natural language description
        Returns:
            src_tensor: torch.Tensor - preprocessed input tensor
            src_length: int - length of the input sequence
        """
        # Tokenize
        tokens = tokenize_nl(natural_language_query)
        
        # Add special tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.src_vocab.word2idx:
                indices.append(self.src_vocab.word2idx[token])
            else:
                indices.append(self.src_vocab.word2idx['<unk>'])
        
        # Create tensor
        src_tensor = torch.LongTensor(indices).unsqueeze(1).to(self.device)  # [seq_len, 1]
        src_length = [len(indices)]
        
        return src_tensor, src_length
    
    def postprocess_output(self, output_indices):
        """
        Convert output indices to readable command
        Args:
            output_indices: list of int - predicted token indices
        Returns:
            command: str - predicted command
        """
        tokens = []
        for idx in output_indices:
            if idx in self.trg_vocab.idx2word:
                token = self.trg_vocab.idx2word[idx]
                if token in ['<eos>', '<pad>']:
                    break
                if token not in ['<sos>', '<unk>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def predict(self, natural_language_query, use_beam_search=True):
        """
        Predict command from natural language query
        Args:
            natural_language_query: str - natural language description
            use_beam_search: bool - whether to use beam search or greedy decoding
        Returns:
            dict with prediction results
        """
        # Preprocess input
        src_tensor, src_length = self.preprocess_input(natural_language_query)
        
        self.model.eval()
        with torch.no_grad():
            if use_beam_search:
                # Use beam search
                predicted_indices, score = self.beam_decoder.beam_search(src_tensor, src_length)
                
                if predicted_indices:
                    predicted_command = self.postprocess_output(predicted_indices)
                    confidence = float(score)  # Log probability
                else:
                    predicted_command = ""
                    confidence = float('-inf')
            else:
                # Use greedy decoding
                sos_token = self.trg_vocab.word2idx['<sos>']
                eos_token = self.trg_vocab.word2idx['<eos>']
                
                predicted_tensor, attention_weights = self.model.inference(
                    src_tensor, src_length, config.MAX_LEN, sos_token, eos_token
                )
                
                if predicted_tensor.size(0) > 0:
                    predicted_indices = predicted_tensor[:, 0].cpu().numpy().tolist()
                    predicted_command = self.postprocess_output(predicted_indices)
                    confidence = 0.0  # Placeholder for greedy decoding
                else:
                    predicted_command = ""
                    confidence = 0.0
        
        return {
            'input': natural_language_query,
            'prediction': predicted_command,
            'confidence': confidence,
            'method': 'beam_search' if use_beam_search else 'greedy'
        }
    
    def predict_batch(self, queries, use_beam_search=True):
        """
        Predict commands for a batch of queries
        Args:
            queries: list of str - natural language queries
            use_beam_search: bool - whether to use beam search
        Returns:
            list of prediction results
        """
        results = []
        for query in queries:
            result = self.predict(query, use_beam_search)
            results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(description='Command Generation Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (if not in checkpoint)')
    parser.add_argument('--input', type=str, default=None,
                       help='Single input query for prediction')
    parser.add_argument('--input_file', type=str, default=None,
                       help='File containing input queries (one per line)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for predictions (JSON format)')
    parser.add_argument('--beam_search', action='store_true', default=True,
                       help='Use beam search (default: True)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize inference
    print("Loading model...")
    inference = CommandInference(args.model_path, args.vocab_path)
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE COMMAND GENERATION")
        print("="*60)
        print("Enter natural language descriptions to generate commands.")
        print("Type 'quit' to exit.")
        print("Type 'beam' to toggle beam search on/off.")
        print("-"*60)
        
        use_beam = args.beam_search
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'beam':
                    use_beam = not use_beam
                    print(f"Beam search {'enabled' if use_beam else 'disabled'}")
                    continue
                elif not user_input:
                    continue
                
                result = inference.predict(user_input, use_beam_search=use_beam)
                
                print(f"\nInput:      {result['input']}")
                print(f"Command:    {result['prediction']}")
                if use_beam:
                    print(f"Confidence: {result['confidence']:.4f}")
                print(f"Method:     {result['method']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    elif args.input:
        # Single input prediction
        result = inference.predict(args.input, use_beam_search=args.beam_search)
        
        print("\nPrediction Result:")
        print(f"Input:      {result['input']}")
        print(f"Command:    {result['prediction']}")
        if args.beam_search:
            print(f"Confidence: {result['confidence']:.4f}")
        print(f"Method:     {result['method']}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump([result], f, indent=2)
            print(f"\nResult saved to {args.output_file}")
    
    elif args.input_file:
        # Batch prediction from file
        print(f"Loading queries from {args.input_file}...")
        
        with open(args.input_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(queries)} queries...")
        results = inference.predict_batch(queries, use_beam_search=args.beam_search)
        
        # Print results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Input:      {result['input']}")
            print(f"   Command:    {result['prediction']}")
            if args.beam_search:
                print(f"   Confidence: {result['confidence']:.4f}")
        
        # Save results
        output_file = args.output_file or 'batch_predictions.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    else:
        # Demo mode with sample queries
        print("\nDemo Mode - Sample Predictions:")
        print("-" * 40)
        
        sample_queries = [
            "List all files in the current directory",
            "Find all Python files in the project",
            "Show the last 10 lines of a log file",
            "Create a new directory called 'backup'",
            "Run a Docker container with Ubuntu image",
            "Stop all running Docker containers",
            "Copy all text files to backup directory",
            "Search for 'error' in all log files",
            "Show disk usage of current directory",
            "Download a file from a URL"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            result = inference.predict(query, use_beam_search=args.beam_search)
            print(f"\n{i}. Query: {result['input']}")
            print(f"   Command: {result['prediction']}")

if __name__ == "__main__":
    main()