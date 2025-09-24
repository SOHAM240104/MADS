"""
Seq2Seq Transformer model for command generation
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(config.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        memory = self.encoder(
            src,
            src_mask,
            src_padding_mask
        )
        return memory


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_layers
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tgt = self.embedding(tgt) * math.sqrt(config.d_model)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_padding_mask,
            memory_padding_mask
        )
        output = self.output_layer(output)
        return output


class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        memory = self.encoder(src, src_mask, src_padding_mask)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_padding_mask,
            memory_padding_mask
        )
        return output

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, tgt_mask)

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        max_len: int,
        start_symbol: int
    ) -> torch.Tensor:
        """Generate a target sequence using greedy decoding."""
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).long()
        
        for i in range(max_len - 1):
            memory = memory.to(src.device)
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            out = self.decode(ys, memory, tgt_mask)
            prob = F.softmax(out[:, -1], dim=-1)
            next_word = torch.argmax(prob, dim=1)
            
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
            
            if next_word == config.eos_token_id:
                break
        
        return ys
    
    def beam_search(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        max_len: int,
        start_symbol: int,
        beam_size: int = 5,
        top_k: int = 1
    ) -> List[torch.Tensor]:
        """Generate k best target sequences using beam search."""
        memory = self.encode(src, src_mask)
        
        # Initialize beam
        beam = [(torch.ones(1, 1).fill_(start_symbol).type_as(src).long(), 0.0)]
        
        for _ in range(max_len - 1):
            candidates = []
            
            for seq, score in beam:
                if seq[0, -1].item() == config.eos_token_id:
                    candidates.append((seq, score))
                    continue
                
                memory = memory.to(src.device)
                tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(src.device)
                out = self.decode(seq, memory, tgt_mask)
                probs = F.log_softmax(out[:, -1], dim=-1)
                
                values, indices = probs[0].topk(beam_size)
                for prob, idx in zip(values, indices):
                    new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src).fill_(idx.item())], dim=1)
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]
            
            # Early stopping if all beams end with EOS
            if all(seq[0, -1].item() == config.eos_token_id for seq, _ in beam):
                break
        
        # Return top_k sequences
        beam.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in beam[:top_k]]