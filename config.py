# config.py

import torch, os

# Dataset paths
DATA_CSV = "/mnt/data/train_v2_drcat_02.csv"
TEXT_COL = "text"
LABEL_COL = "label"

# Model configuration
BACKBONES = [
    "microsoft/deberta-v3-base",
    "albert-base-v2",
    "distilbert-base-uncased"
]

MAX_LEN = 256
OVERLAP = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 6
LEARNING_RATE = 1e-4
FUSION_HIDDEN = 512
EMB_DIR = "./embeddings_cache"
os.makedirs(EMB_DIR, exist_ok=True)
