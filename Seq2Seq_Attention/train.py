import config
import dataset
import model
import engine
import utils
import time
from torchtext.data import BucketIterator
import torch.optim as optim
import torch.nn as nn
import torch
import math
import random
import numpy as np
import argparse
from torchtext.data.metrics import bleu_score
import sys
sys.path.append('../clai/utils/metric')
import metric_utils
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument("--action", 
	type=str, 
	default='train', 
	help="whether to train or test")

args = parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def run():
    Seed = 1234
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.backends.cudnn.deterministic = True
    
    train, valid, test, SRC, TRG = dataset.create_dataset()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        sort_key=lambda x: len(x.source),
        batch_size=config.BATCH_SIZE,
        device=config.device
        )
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    
    ENC_EMB_DIM = config.ENCODER_EMBEDDING_DIMENSION
    DEC_EMB_DIM = config.DECODER_EMBEDDING_DIMENSION
    HID_DIM = config.LSTM_HIDDEN_DIMENSION
    N_LAYERS = config.LSTM_LAYERS
    ENC_DROPOUT = config.ENCODER_DROPOUT
    DEC_DROPOUT = config.DECODER_DROPOUT

    attn = model.Attention(HID_DIM, HID_DIM)
    enc = model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT)
    dec = model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, attn)

    model_rnn = model.Seq2Seq(enc, dec, config.device).to(config.device)

    # Add L2 regularization (weight decay)
    optimizer = optim.Adam(model_rnn.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR
    )

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    if(args.action=='train'):
        model_rnn.apply(utils.init_weights)

        best_valid_loss = float('inf')
        best_epoch = 0
        early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
        
        # Clear results file
        with open(config.RESULTS_SAVE_FILE, 'w') as f:
            f.write("Training Results with Early Stopping and LR Scheduling\n")
            f.write("=" * 50 + "\n")

        for epoch in range(config.N_EPOCHS):    
            start_time = time.time()

            train_loss = engine.train_fn(model_rnn, train_iterator, optimizer, criterion, config.CLIP)
            valid_loss = engine.evaluate_fn(model_rnn, valid_iterator, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            # Learning rate scheduling
            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Save best model based on validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                torch.save(model_rnn.state_dict(), config.BEST_MODEL_SAVE_FILE)
                print(f'New best model saved! Epoch {epoch+1}, Val Loss: {valid_loss:.3f}')

            # Save current model
            torch.save(model_rnn.state_dict(), config.MODEL_SAVE_FILE)

            # Log results
            with open(config.RESULTS_SAVE_FILE, 'a') as f:
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s', file=f)
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', file=f)
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}', file=f)
                print(f'\t Learning Rate: {current_lr:.6f}', file=f)
                print(f'\t Best Val Loss: {best_valid_loss:.3f} (Epoch {best_epoch})', file=f)
                print('-' * 50, file=f)

            # Print to console for monitoring
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Learning Rate: {current_lr:.6f}')
            print(f'\t Best Val Loss: {best_valid_loss:.3f} (Epoch {best_epoch})')

            # Early stopping check
            if early_stopping(valid_loss):
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Final summary
        with open(config.RESULTS_SAVE_FILE, 'a') as f:
            print(f'\nTraining completed!', file=f)
            print(f'Best validation loss: {best_valid_loss:.3f} at epoch {best_epoch}', file=f)
            print(f'Total epochs trained: {epoch+1}', file=f)
    
    elif(args.action=='test'):  
        # Load the best model for testing
        try:
            model_rnn.load_state_dict(torch.load(config.BEST_MODEL_SAVE_FILE))
            print(f"Loaded best model from {config.BEST_MODEL_SAVE_FILE}")
        except:
            model_rnn.load_state_dict(torch.load(config.MODEL_SAVE_FILE))
            print(f"Loaded current model from {config.MODEL_SAVE_FILE}")
            
        loss, target, output = engine.test_fn(model_rnn, test_iterator, criterion, SRC, TRG)
        bl = bleu_score(output, target, max_n=1, weights=[1])
        met = 0
        
        for z in range(len(output)):
            out = ' '.join(output[z][y] for y in range(min(10, len(output[z]))))
            tar = ' '.join(y for y in target[z])
            met = met + metric_utils.compute_metric(out, 1.0, tar) 
            
        with open(config.TEST_RESULTS_FILE, 'w') as f:
            print(f'Test bleu :, {bl*100}, Test PPL: {math.exp(loss):7.3f}', 'Metric:', met/len(output), file=f)
            print(f'Model used: Best model from training', file=f)

    elif(args.action=='save_vocab'):
        print('Source Vocab Length', len(SRC.vocab))
        print('Target vocab length', len(TRG.vocab))
        s1 = '\n'.join(k for k in SRC.vocab.itos)
        s2 = '\n'.join(k for k in TRG.vocab.itos)
        with open('NL_vocabulary.txt', 'w') as f:
            f.write(s1)
        with open('Bash_vocabulary.txt', 'w') as f:
            f.write(s2)

if __name__=='__main__':
    run()

