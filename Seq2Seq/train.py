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

sys.path.append('..')
sys.path.append('../clai/utils/metric')

import  metric_utils

parser = argparse.ArgumentParser()
parser.add_argument("--action", 
	type=str, 
	default='train', 
	help="whether to train or test")
parser.add_argument("--text",
    type=str,
    default="",
    help="input natural language for inference when action is 'infer'")

args = parser.parse_args()



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
        sort_key=lambda x: len(x.nl),
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

    enc = model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model_rnn = model.Seq2seq(enc, dec, config.device).to(config.device)
    model_rnn.apply(utils.init_weights)

    optimizer = optim.Adam(model_rnn.parameters(), lr=config.LEARNING_RATE)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    best_valid_loss = float('inf')

    if(args.action=='train'):
        for epoch in range(config.N_EPOCHS):    
            start_time = time.time()

            train_loss = engine.train_fn(model_rnn, train_iterator, optimizer, criterion, config.CLIP)
            valid_loss = engine.evaluate_fn(model_rnn, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_rnn.state_dict(), config.MODEL_SAVE_FILE)

            with open(config.RESULTS_SAVE_FILE, 'a') as f:
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s', file=f)
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', file=f)
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}', file=f)

    elif(args.action=='test'):
        model_rnn.load_state_dict(torch.load(config.TEST_MODEL))      
        loss, target, output = engine.test_fn(model_rnn, test_iterator, criterion, SRC, TRG)

        # Filter out empty predictions/references and format refs as list-of-lists
        pairs = [(o, t) for o, t in zip(output, target) if len(o) > 0 and len(t) > 0]
        if len(pairs) == 0:
            bl = 0.0
            avg_metric = 0.0
        else:
            pred_corpus = [o for o, _ in pairs]
            ref_corpus = [[t] for _, t in pairs]
            bl = bleu_score(pred_corpus, ref_corpus, max_n=1, weights=[1])

            met = 0.0
            for o, t in pairs:
                out = ' '.join(o)
                tar = ' '.join(t)
                met += metric_utils.compute_metric(out, 1.0, tar)
            avg_metric = met / len(pairs)

        with open(config.TEST_RESULTS_FILE, 'a') as f:
            print(f'Test bleu :, {bl*100}, Test PPL: {math.exp(loss):7.3f}', 'Metric:', avg_metric, file=f)

    elif(args.action=='infer'):
        # Load best model
        model_rnn.load_state_dict(torch.load(config.TEST_MODEL))
        model_rnn.eval()

        # Helper to translate a sentence using greedy decoding
        def translate_sentence(sentence: str, max_len: int = 50):
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

        input_text = args.text.strip()
        if not input_text:
            input_text = "create a directory"

        pred_tokens = translate_sentence(input_text)
        # Strip special tokens and <unk>
        cleaned = [t for t in pred_tokens if t not in {TRG.init_token, TRG.eos_token, TRG.pad_token, '<unk>'}]
        print('Input :', input_text)
        print('Output:', ' '.join(cleaned))

if __name__=='__main__':
    run()