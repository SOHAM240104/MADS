import torch

# Use MPS (Metal Performance Shaders) for Apple Silicon, fallback to CPU if not available
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
BATCH_SIZE = 8
LEARNING_RATE=0.0001
N_EPOCHS = 10
CLIP = 1
BASH_DATASET_PATH = '/Users/mohamedaamir/Documents/MADS/data/bash_dataset.json'
DOCKER_DATASET_PATH = '/Users/mohamedaamir/Documents/MADS/data/docker_dataset.json'
GLOVE_PATH = '/Users/mohamedaamir/Documents/MADS/glove.840B.300d.txt'
TRAIN_SPLIT=0.8
TEST_SPLIT=0.1
VALID_SPLIT=0.1
ENCODER_EMBEDDING_DIMENSION=300
DECODER_EMBEDDING_DIMENSION=300
LSTM_HIDDEN_DIMENSION = 512
LSTM_LAYERS=2
ENCODER_DROPOUT=0.5
DECODER_DROPOUT=0.5
LEARNING_RATE=0.001
MAX_LEN=70

# Advanced Training Parameters
WARMUP_STEPS = 1000
LABEL_SMOOTHING = 0.1
TEACHER_FORCING_RATIO = 0.8
BEAM_SIZE = 5

# Model save paths
MODEL_SAVE_FILE='best_model_seq2seq_attention.bin'
CHECKPOINT_DIR='checkpoints'
LOG_DIR='logs'

# Results files
RESULTS_SAVE_FILE='results_attention_final.txt'
TEST_RESULTS_FILE='test_attention_final.txt'
TRAINING_HISTORY_FILE='training_history.json'

# Evaluation parameters
EVAL_EVERY_N_EPOCHS = 3
SAVE_TOP_K_MODELS = 3

# Early stopping
EARLY_STOPPING_PATIENCE = 7
MIN_DELTA = 0.001

# Professional AI settings
USE_LABEL_SMOOTHING = True
USE_BEAM_SEARCH = True
USE_GRADIENT_CLIPPING = True
USE_LEARNING_RATE_SCHEDULING = True
USE_EARLY_STOPPING = True

