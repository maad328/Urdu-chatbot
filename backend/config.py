"""
Configuration file for the Urdu Chatbot
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model Configuration
EMBED_DIM = 512
NUM_HEADS = 4
FF_DIM = 2048
DROPOUT = 0.4
ENC_LAYERS = 3
DEC_LAYERS = 3
MAX_SEQ_LEN = 128

# Device configuration
DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

# File paths - Updated to match your current structure
TOKENIZER_PATH = str(BASE_DIR / "tokenizer" / "urdu_tokenizer.model")
MODEL_PATH = str(BASE_DIR / "model" / "best_finetuned_epoch30_loss0.3141.pt")

# Generation settings
DEFAULT_MAX_LENGTH = 100
MAX_GENERATION_LENGTH = 200

