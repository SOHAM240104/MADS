"""
Training script for Seq2Seq Transformer
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from config import config
from model_new import Seq2SeqTransformer
from dataset import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    pad_idx: int
) -> float:
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        # Get batch data
        src = batch['input_ids'].to(config.device)
        tgt = batch['labels'].to(config.device)
        tgt_input = tgt[:, :-1]  # Remove last token for input
        
        # Create masks
        src_key_padding_mask = ~batch['attention_mask'].bool().to(config.device)  # Convert to padding mask
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(config.device)
        src_mask = None  # Let the model handle padding via key_padding_mask
        
        # Forward pass
        logits = model(
            src=src,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_key_padding_mask
        )
        
        # Calculate loss
        optimizer.zero_grad()
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt[:, 1:].reshape(-1)  # Remove first token (SOS) for target
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    pad_idx: int
) -> float:
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(config.device)
            tgt = batch['labels'].to(config.device)
            tgt_input = tgt[:, :-1]
            
            src_key_padding_mask = ~batch['attention_mask'].bool().to(config.device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(config.device)
            
            logits = model(
                src=src,
                tgt=tgt_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_padding_mask=src_key_padding_mask
            )
            
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train():
    # Create model directory if it doesn't exist
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config.tokenizer)
    logger.info("Created data loaders")
    
    # Initialize model
    model = Seq2SeqTransformer().to(config.device)
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint = None
    
    # Load checkpoint if it exists
    if config.model_path.exists():
        logger.info(f"Loading checkpoint from {config.model_path}")
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        logger.info(f"Loaded checkpoint from epoch {start_epoch-1} with validation loss {best_val_loss:.4f}")
    logger.info("Model initialized")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Load optimizer state if checkpoint exists
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    num_training_steps = len(train_loader) * config.n_epochs
    warmup_steps = config.warmup_steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 3
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.n_epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch + 1,
            config.pad_token_id
        )
        
        val_loss = validate(model, val_loader, config.pad_token_id)
        
        logger.info(f"Epoch {epoch + 1}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, config.model_path)
            
            logger.info("Saved best model")
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break
    
    logger.info("Training completed")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()


if __name__ == "__main__":
    train()

