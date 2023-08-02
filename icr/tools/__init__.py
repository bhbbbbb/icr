import os
import numpy as np

def is_on_kaggle():
    """determine whether the environment is on kaggle."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


def seed_everything(seed=42):
    import random
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def alpha_to_class(alpha: np.ndarray):
    return (alpha != 0).astype(int)
