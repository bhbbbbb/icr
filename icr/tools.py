import os
import numpy as np
from sklearn.metrics import log_loss, confusion_matrix
from termcolor import colored

def is_on_kaggle():
    """determine whether the environment is on kaggle."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

def balanced_log_loss(y_pred: np.ndarray, y_true: np.ndarray):

    class_weights = 1 / np.array([(y_true == 0).sum(), (y_true == 1).sum()])

    return log_loss(y_true, y_pred, sample_weight=class_weights[y_true], eps=1e-15)

def precision_recall(y_pred: np.ndarray, y_true: np.ndarray):

    mat = confusion_matrix(y_true, (y_pred > .5).astype(int))
    tn, fp, fn, tp = mat.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    wrongs = fp + fn
    return precision, recall, wrongs

def color_precision_recall(precision, recall):
    def get_color(my, other):
        if my == 1.:
            return 'green'
        return 'green' if my > other else 'red'
    precision_str = colored(f'{precision:.3f}', get_color(precision, recall))
    recall_str = colored(f'{recall:.3f}', get_color(recall, precision))
    return precision_str, recall_str

def seed_everything(seed=42):
    import random
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
