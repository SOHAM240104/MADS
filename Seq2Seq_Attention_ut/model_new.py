import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM Encoder with multi-layer support
    """
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout_prob):
        super(BiLSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=False
        )
        
        # Projection layer to combine forward and backward hidden states
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: [seq_len, batch_size]
            src_lengths: [batch_size] - lengths of source sequences
        Returns:
            outputs: [seq_len, batch_size, hidden_dim * 2] - encoder outputs
            hidden: [batch_size, hidden_dim] - final hidden state
            cell: [batch_size, hidden_dim] - final cell state
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))  # [seq_len, batch_size, emb_dim]
        
        # Pack padded sequence if lengths are provided
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, enforce_sorted=False)
        
        # Bi-directional LSTM
        lstm_outputs, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if we packed
        if src_lengths is not None:
            lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs)
        
        # lstm_outputs: [seq_len, batch_size, hidden_dim * 2]
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # cell: [num_layers * 2, batch_size, hidden_dim]
        
        # Combine forward and backward final hidden states
        # Take the last layer's hidden and cell states
        forward_hidden = hidden[-2, :, :]  # Last forward layer
        backward_hidden = hidden[-1, :, :]  # Last backward layer
        forward_cell = cell[-2, :, :]
        backward_cell = cell[-1, :, :]
        
        # Combine forward and backward states
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        combined_cell = torch.cat([forward_cell, backward_cell], dim=1)
        
        # Project to decoder hidden size
        final_hidden = torch.tanh(self.hidden_projection(combined_hidden))
        final_cell = torch.tanh(self.cell_projection(combined_cell))
        
        return lstm_outputs, final_hidden, final_cell

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        
        self.encoder_projection = nn.Linear(encoder_hidden_dim * 2, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [seq_len, batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, seq_len] - mask for padded positions
        Returns:
            attention_weights: [batch_size, seq_len]
            context_vector: [batch_size, encoder_hidden_dim * 2]
        """
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        # Project encoder outputs
        encoder_projected = self.encoder_projection(encoder_outputs)  # [seq_len, batch_size, attention_dim]
        
        # Project decoder hidden state and expand
        decoder_projected = self.decoder_projection(decoder_hidden).unsqueeze(0)  # [1, batch_size, attention_dim]
        decoder_projected = decoder_projected.expand(seq_len, -1, -1)  # [seq_len, batch_size, attention_dim]
        
        # Compute attention scores
        attention_scores = self.attention_vector(
            torch.tanh(encoder_projected + decoder_projected)
        ).squeeze(2)  # [seq_len, batch_size]
        
        # Transpose for softmax
        attention_scores = attention_scores.transpose(0, 1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector
        encoder_outputs_transposed = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, encoder_hidden_dim * 2]
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs_transposed
        ).squeeze(1)  # [batch_size, encoder_hidden_dim * 2]
        
        return attention_weights, context_vector

class LSTMDecoder(nn.Module):
    """
    Unidirectional LSTM Decoder with Bahdanau Attention
    """
    def __init__(self, output_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, 
                 attention_dim, dropout_prob):
        super(LSTMDecoder, self).__init__()
        
        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, attention_dim)
        
        # LSTM cell - input is embedding + context vector
        self.lstm = nn.LSTMCell(emb_dim + encoder_hidden_dim * 2, decoder_hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim * 2 + emb_dim, 
            output_dim
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        """
        Args:
            input_token: [batch_size] - current input token
            hidden: [batch_size, decoder_hidden_dim] - previous hidden state
            cell: [batch_size, decoder_hidden_dim] - previous cell state
            encoder_outputs: [seq_len, batch_size, encoder_hidden_dim * 2]
            mask: [batch_size, seq_len] - mask for encoder outputs
        Returns:
            output: [batch_size, output_dim] - output distribution
            hidden: [batch_size, decoder_hidden_dim] - new hidden state
            cell: [batch_size, decoder_hidden_dim] - new cell state
            attention_weights: [batch_size, seq_len] - attention weights
        """
        # Embedding
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, emb_dim]
        
        # Compute attention
        attention_weights, context_vector = self.attention(hidden, encoder_outputs, mask)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context_vector], dim=1)  # [batch_size, emb_dim + encoder_hidden_dim * 2]
        
        # LSTM step
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Output projection
        output_input = torch.cat([hidden, context_vector, embedded], dim=1)
        output = self.output_projection(output_input)  # [batch_size, output_dim]
        
        return output, hidden, cell, attention_weights

class Seq2SeqWithAttention(nn.Module):
    """
    Sequence-to-Sequence model with Bi-directional LSTM Encoder and LSTM Decoder with Bahdanau Attention
    """
    def __init__(self, src_vocab_size, trg_vocab_size, encoder_emb_dim, decoder_emb_dim,
                 encoder_hidden_dim, decoder_hidden_dim, encoder_num_layers, attention_dim,
                 encoder_dropout, decoder_dropout, device):
        super(Seq2SeqWithAttention, self).__init__()
        
        self.device = device
        self.trg_vocab_size = trg_vocab_size
        
        # Encoder
        self.encoder = BiLSTMEncoder(
            input_dim=src_vocab_size,
            emb_dim=encoder_emb_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout_prob=encoder_dropout
        )
        
        # Decoder
        self.decoder = LSTMDecoder(
            output_dim=trg_vocab_size,
            emb_dim=decoder_emb_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            dropout_prob=decoder_dropout
        )
        
    def create_mask(self, src, src_lengths):
        """Create mask for padded positions"""
        batch_size = src.size(1)
        max_len = src.size(0)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        
        for i, length in enumerate(src_lengths):
            mask[i, :length] = 1
            
        return mask
        
    def forward(self, src, trg, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [seq_len, batch_size] - source sequences
            trg: [seq_len, batch_size] - target sequences
            src_lengths: [batch_size] - source sequence lengths
            teacher_forcing_ratio: float - probability of using teacher forcing
        Returns:
            outputs: [trg_len, batch_size, trg_vocab_size] - decoder outputs
            attention_weights: list of attention weights for each time step
        """
        batch_size = src.size(1)
        trg_len = trg.size(0)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Create mask for attention
        mask = self.create_mask(src, src_lengths) if src_lengths is not None else None
        
        # Initialize outputs
        outputs = torch.zeros(trg_len, batch_size, self.trg_vocab_size, device=self.device)
        attention_weights_list = []
        
        # First input is <sos> token
        input_token = trg[0, :]  # [batch_size]
        
        # Decode step by step
        for t in range(1, trg_len):
            # Decoder step
            output, hidden, cell, attention_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            # Store output and attention weights
            outputs[t] = output
            attention_weights_list.append(attention_weights)
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get next input
            if teacher_force:
                input_token = trg[t, :]  # Use ground truth
            else:
                input_token = output.argmax(dim=1)  # Use model prediction
        
        return outputs, attention_weights_list
    
    def inference(self, src, src_lengths, max_len, sos_token, eos_token):
        """
        Inference without teacher forcing
        """
        self.eval()
        batch_size = src.size(1)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # Create mask
            mask = self.create_mask(src, src_lengths) if src_lengths is not None else None
            
            # Initialize
            outputs = []
            attention_weights_list = []
            input_token = torch.LongTensor([sos_token] * batch_size).to(self.device)
            
            for t in range(max_len):
                # Decoder step
                output, hidden, cell, attention_weights = self.decoder(
                    input_token, hidden, cell, encoder_outputs, mask
                )
                
                # Get predicted token
                predicted_token = output.argmax(dim=1)
                outputs.append(predicted_token)
                attention_weights_list.append(attention_weights)
                
                # Check for end of sequence
                if (predicted_token == eos_token).all():
                    break
                    
                input_token = predicted_token
            
            # Stack outputs
            if outputs:
                outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size]
            else:
                outputs = torch.empty(0, batch_size, dtype=torch.long, device=self.device)
        
        return outputs, attention_weights_list

def create_model(src_vocab_size, trg_vocab_size, device):
    """Create and initialize the model"""
    model = Seq2SeqWithAttention(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        encoder_emb_dim=config.ENCODER_EMBEDDING_DIMENSION,
        decoder_emb_dim=config.DECODER_EMBEDDING_DIMENSION,
        encoder_hidden_dim=config.LSTM_HIDDEN_DIMENSION,
        decoder_hidden_dim=config.LSTM_HIDDEN_DIMENSION,
        encoder_num_layers=config.LSTM_LAYERS,
        attention_dim=config.LSTM_HIDDEN_DIMENSION,
        encoder_dropout=config.ENCODER_DROPOUT,
        decoder_dropout=config.DECODER_DROPOUT,
        device=device
    ).to(device)
    
    return model

def init_weights(model):
    """Initialize model weights"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)