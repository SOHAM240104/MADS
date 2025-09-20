import torch

# Use MPS (Metal Performance Shaders) for Apple Silicon, fallback to CPU if not available
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Training parameters
BATCH_SIZE = 300
LEARNING_RATE = 0.001
N_EPOCHS = 100  # Increased epochs for early stopping
CLIP = 1

# Data paths
TRAIN_PATH = '../data/train.csv'
VALID_PATH = '../data/valid.csv'
TEST_PATH = '../data/test.csv'
X_FORMAT = 'csv'
GLOVE_PATH = '../glove.840B.300d.txt'

# Data splits
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
VALID_SPLIT = 0.1

# Model architecture
ENCODER_EMBEDDING_DIMENSION = 300
DECODER_EMBEDDING_DIMENSION = 300
LSTM_HIDDEN_DIMENSION = 512
LSTM_LAYERS = 2

# Regularization and overfitting prevention
ENCODER_DROPOUT = 0.3  # Reduced from 0.5
DECODER_DROPOUT = 0.3  # Reduced from 0.5
WEIGHT_DECAY = 1e-5    # L2 regularization
TEACHER_FORCING_RATIO = 0.5

# Early stopping parameters
PATIENCE = 10  # Number of epochs to wait before early stopping
MIN_DELTA = 0.001  # Minimum improvement required

# Learning rate scheduling
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Model save files
MODEL_SAVE_FILE = 'model_seq2seq_attention.bin'
BEST_MODEL_SAVE_FILE = 'best_model_seq2seq_attention.bin'
RESULTS_SAVE_FILE = 'results_attention.txt'
TEST_RESULTS_FILE = 'test_attention.txt'


