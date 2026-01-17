import os
import torch


class Config:
    """Configuration class for LPR training."""
    
    # Data paths
    DATA_ROOT = "data/train"
    VAL_SPLIT_FILE = "val_tracks.json"
    TEST_SPLIT_FILE = "test_tracks.json"
    
    # Image settings
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    
    # Character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

    # Evaluation-only normalization
    # If set, these characters are removed from BOTH prediction and ground-truth
    # strings before computing exact-match val_acc (training is unaffected).
    # Example: "-" to ignore printed hyphens that are not part of labels.
    EVAL_STRIP_CHARS = ""
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    SEED = 42
    NUM_WORKERS = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Character mappings (computed from CHARS)
    CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
