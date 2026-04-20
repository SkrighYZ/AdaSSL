
import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_sizes(s):
    return [int(x) for x in s.split('-')]

def find_latest_checkpoint(save_dir):
    # Find the latest checkpoint in save_dir.
    ckpt_files = list(save_dir.glob('ckpt_*.pt'))
    if not ckpt_files:
        return None
    # Extract step numbers and find the latest
    ckpt_steps = [(int(f.stem.split('_')[1]), f) for f in ckpt_files]
    _, latest_ckpt = max(ckpt_steps, key=lambda x: x[0])
    return latest_ckpt