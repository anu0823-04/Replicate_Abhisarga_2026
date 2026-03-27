"""
utils.py — Helper functions for replication
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"[Seed set to {seed}]")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def log(msg: str):
    """Simple timestamped logger."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
