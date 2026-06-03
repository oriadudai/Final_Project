import os
import torch

DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#data parameters:
FS = 125 #Sampling Rate
WINDOW_SIZE = 10
SEQ_LEN = FS * WINDOW_SIZE
#Model Hyperparameters
INPUT_CHANNELS = 1
HIDDEN_SIZE = 64 # number if neurons in the BiLSTM

#  number of the DC blocks-BiLSTM
NUM_BLOCKS = 5
# Training Hyperparameters:
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
#device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#printing the configuration for the first setup 
if __name__ == "__main__":
    print("=== ReHeartNet Configuration Initialized ===")
    print(f"Signal Sequence Length: {SEQ_LEN} samples")
    print(f"Training Environment Device: {DEVICE}")
