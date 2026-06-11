import os
import random
import numpy as np
import tensorflow as tf

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────
# Assumes script is run from the root of the repository
DATA_ROOT   = 'data/chest_xray'
OUTPUT_DIR  = 'outputs'
MODEL_DIR   = 'outputs/models'

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────
IMG_SIZE      = 224          
BATCH_SIZE    = 16           
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20
PATIENCE      = 5
STAGE1_LR     = 1e-3
STAGE2_LR     = 1e-5
DROPOUT       = 0.4
DENSE_UNITS   = 512
THRESHOLD     = 0.5
CLASS_NAMES   = ['NORMAL', 'PNEUMONIA']