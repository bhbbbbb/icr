import os
import numpy as np
from sklearn.metrics import log_loss

def is_on_kaggle():
    """determine whether the environment is on kaggle."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

def balanced_log_loss(y_pred: np.ndarray, y_true: np.ndarray):

    class_weights = 1 / np.array([(y_true == 0).sum(), (y_true == 1).sum()])

    return log_loss(y_true, y_pred, sample_weight=class_weights[y_true], eps=1e-15)

def seed_everything(seed=42):
    import random
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
