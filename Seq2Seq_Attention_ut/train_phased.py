import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
import logging
from datetime import datetime
import json

import config
from dataset_new import (create_bash_only_dataset, create_dataset_and_vocabs, 
                        get_data_loaders)
from model_new import create_model, init_weights, count_parameters
from evaluation import evaluate_model, print_evaluation_results

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, vocab_info, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'vocab_info': vocab_info,
        'model_config': {
            'src_vocab_size': model.encoder.embedding.num_embeddings,
            'trg_vocab_size': model.decoder.output_dim,
            'encoder_emb_dim': config.ENCODER_EMBEDDING_DIMENSION,
            'decoder_emb_dim': config.DECODER_EMBEDDING_DIMENSION,
            'encoder_hidden_dim': config.LSTM_HIDDEN_DIMENSION,
            'decoder_hidden_dim': config.LSTM_HIDDEN_DIMENSION,
            'encoder_num_layers': config.LSTM_LAYERS,
            'attention_dim': config.LSTM_HIDDEN_DIMENSION,
            'encoder_dropout': config.ENCODER_DROPOUT,
            'decoder_dropout': config.DECODER_DROPOUT
        }
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def train_epoch(model, data_loader, optimizer, criterion, device, clip_grad_norm, logger):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(data_loader):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        src_lengths = batch['src_lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(src, trg, src_lengths, teacher_forcing_ratio=0.8)
        
        # Calculate loss
        outputs_flat = outputs[1:].reshape(-1, outputs.shape[-1])
        trg_flat = trg[1:].reshape(-1)
        loss = criterion(outputs_flat, trg_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(f'  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')
    
    return epoch_loss / num_batches

def validate_epoch(model, data_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            src_lengths = batch['src_lengths']
            
            # Forward pass
            outputs, _ = model(src, trg, src_lengths, teacher_forcing_ratio=0)
            
            # Calculate loss
            outputs_flat = outputs[1:].reshape(-1, outputs.shape[-1])
            trg_flat = trg[1:].reshape(-1)
            loss = criterion(outputs_flat, trg_flat)
            
            epoch_loss += loss.item()
            num_batches += 1
    
    return epoch_loss / num_batches

def train_phase(model, train_loader, valid_loader, test_loader, trg_vocab, 
                phase_name, num_epochs, device, logger, checkpoint_dir):
    """Train a single phase"""
    logger.info(f"Starting {phase_name}...")
    
    # Setup training components
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    # Training metrics
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    max_patience = 7
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                               device, config.CLIP, logger)
        
        # Validation
        valid_loss = validate_epoch(model, valid_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(valid_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        epoch_time = time.time() - start_time
        
        logger.info(f'{phase_name} Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Valid Loss: {valid_loss:.4f}')
        logger.info(f'  Time: {epoch_time:.2f}s')
        logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'{phase_name.lower()}_best_model.pt')
            vocab_info = {
                'src_vocab_size': len(trg_vocab.word2idx),
                'trg_vocab_size': len(trg_vocab.word2idx),
                'src_word2idx': {},  # Simplified for this example
                'trg_word2idx': trg_vocab.word2idx,
                'trg_idx2word': trg_vocab.idx2word
            }
            save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, 
                          vocab_info, checkpoint_path)
            logger.info(f'  Saved best model to {checkpoint_path}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Evaluate every few epochs
        if (epoch + 1) % 3 == 0:
            logger.info('Running evaluation...')
            eval_results = evaluate_model(model, test_loader, trg_vocab, device)
            logger.info(f'Test EM Score: {eval_results["overall"]["exact_match"]:.4f}')
            logger.info(f'Test BLEU Score: {eval_results["overall"]["bleu"]:.4f}')
    
    return train_losses, valid_losses, best_valid_loss

def main():
    """Main training function"""
    # Setup
    logger = setup_logging()
    device = config.device
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SEQUENCE-TO-SEQUENCE MODEL WITH BI-LSTM AND ATTENTION")
    logger.info("=" * 80)
    
    # PHASE 1: Train on bash-only data
    logger.info("PHASE 1: Training on bash-only dataset")
    logger.info("-" * 40)
    
    # Load bash-only data
    bash_train, bash_valid, bash_test, bash_src_vocab, bash_trg_vocab = create_bash_only_dataset()
    bash_train_loader, bash_valid_loader, bash_test_loader = get_data_loaders(
        bash_train, bash_valid, bash_test, bash_src_vocab, bash_trg_vocab
    )
    
    # Create model
    model = create_model(bash_src_vocab.n_words, bash_trg_vocab.n_words, device)
    init_weights(model)
    
    logger.info(f"Model created with {count_parameters(model):,} trainable parameters")
    logger.info(f"Bash vocabulary sizes - Source: {bash_src_vocab.n_words}, Target: {bash_trg_vocab.n_words}")
    
    # Train Phase 1
    phase1_epochs = config.N_EPOCHS
    train_losses_p1, valid_losses_p1, best_loss_p1 = train_phase(
        model, bash_train_loader, bash_valid_loader, bash_test_loader, 
        bash_trg_vocab, "PHASE_1", phase1_epochs, device, logger, checkpoint_dir
    )
    
    # Load best Phase 1 model
    phase1_checkpoint_path = os.path.join(checkpoint_dir, 'phase_1_best_model.pt')
    if os.path.exists(phase1_checkpoint_path):
        checkpoint = load_checkpoint(phase1_checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best Phase 1 model")
    
    # Evaluate Phase 1
    logger.info("Evaluating Phase 1 model...")
    eval_results_p1 = evaluate_model(model, bash_test_loader, bash_trg_vocab, device)
    logger.info("Phase 1 Results:")
    print_evaluation_results(eval_results_p1)
    
    # PHASE 2: Fine-tune on combined data
    logger.info("\nPHASE 2: Fine-tuning on combined bash + docker dataset")
    logger.info("-" * 50)
    
    # Load combined data
    combined_train, combined_valid, combined_test, combined_src_vocab, combined_trg_vocab = create_dataset_and_vocabs()
    
    # Create new model for combined vocabulary (if vocabulary size changed significantly)
    if combined_trg_vocab.n_words != bash_trg_vocab.n_words:
        logger.info("Vocabulary size changed, creating new model...")
        new_model = create_model(combined_src_vocab.n_words, combined_trg_vocab.n_words, device)
        
        # Try to transfer weights where possible
        try:
            # Copy encoder weights
            encoder_state = {k: v for k, v in model.encoder.state_dict().items() 
                           if k in new_model.encoder.state_dict() and 
                           v.shape == new_model.encoder.state_dict()[k].shape}
            new_model.encoder.load_state_dict(encoder_state, strict=False)
            
            # Copy decoder weights (except output layer)
            decoder_state = {k: v for k, v in model.decoder.state_dict().items() 
                           if k in new_model.decoder.state_dict() and 
                           v.shape == new_model.decoder.state_dict()[k].shape and
                           'output_projection' not in k}
            new_model.decoder.load_state_dict(decoder_state, strict=False)
            
            logger.info("Successfully transferred compatible weights")
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}. Starting fresh.")
            init_weights(new_model)
        
        model = new_model
    else:
        logger.info("Vocabulary size unchanged, continuing with same model")
    
    logger.info(f"Combined vocabulary sizes - Source: {combined_src_vocab.n_words}, Target: {combined_trg_vocab.n_words}")
    
    # Get combined data loaders
    combined_train_loader, combined_valid_loader, combined_test_loader = get_data_loaders(
        combined_train, combined_valid, combined_test, combined_src_vocab, combined_trg_vocab
    )
    
    # Train Phase 2 (with lower learning rate for fine-tuning)
    original_lr = config.LEARNING_RATE
    config.LEARNING_RATE = original_lr * 0.3  # Reduce learning rate for fine-tuning
    
    phase2_epochs = max(5, config.N_EPOCHS // 2)  # Fewer epochs for fine-tuning
    train_losses_p2, valid_losses_p2, best_loss_p2 = train_phase(
        model, combined_train_loader, combined_valid_loader, combined_test_loader,
        combined_trg_vocab, "PHASE_2", phase2_epochs, device, logger, checkpoint_dir
    )
    
    # Restore original learning rate
    config.LEARNING_RATE = original_lr
    
    # Load best Phase 2 model
    phase2_checkpoint_path = os.path.join(checkpoint_dir, 'phase_2_best_model.pt')
    if os.path.exists(phase2_checkpoint_path):
        checkpoint = load_checkpoint(phase2_checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best Phase 2 model")
    
    # Final evaluation
    logger.info("\nFINAL EVALUATION")
    logger.info("-" * 30)
    eval_results_final = evaluate_model(model, combined_test_loader, combined_trg_vocab, device)
    print_evaluation_results(eval_results_final)
    
    # Save final model
    final_model_path = config.MODEL_SAVE_FILE
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    training_history = {
        'phase1': {
            'train_losses': train_losses_p1,
            'valid_losses': valid_losses_p1,
            'best_valid_loss': best_loss_p1,
            'evaluation': eval_results_p1
        },
        'phase2': {
            'train_losses': train_losses_p2,
            'valid_losses': valid_losses_p2,
            'best_valid_loss': best_loss_p2,
            'evaluation': eval_results_final
        },
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': original_lr,
            'n_epochs_p1': phase1_epochs,
            'n_epochs_p2': phase2_epochs,
            'lstm_hidden_dim': config.LSTM_HIDDEN_DIMENSION,
            'lstm_layers': config.LSTM_LAYERS,
            'encoder_dropout': config.ENCODER_DROPOUT,
            'decoder_dropout': config.DECODER_DROPOUT
        }
    }
    
    with open('training_history.json', 'w') as f:
        # Convert numpy types to python types for JSON serialization
        history_serializable = json.loads(json.dumps(training_history, default=str))
        json.dump(history_serializable, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final Exact Match Score: {eval_results_final['overall']['exact_match']:.4f}")
    logger.info(f"Final BLEU Score: {eval_results_final['overall']['bleu']:.4f}")
    
    if 'bash' in eval_results_final['by_type']:
        bash_em = eval_results_final['by_type']['bash']['exact_match']
        logger.info(f"Bash Commands EM Score: {bash_em:.4f}")
    
    if 'docker' in eval_results_final['by_type']:
        docker_em = eval_results_final['by_type']['docker']['exact_match']
        logger.info(f"Docker Commands EM Score: {docker_em:.4f}")

if __name__ == "__main__":
    main()