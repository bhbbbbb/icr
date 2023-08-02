import numpy as np
from sklearn.metrics import log_loss, confusion_matrix
from termcolor import colored

def balanced_log_loss(y_pred: np.ndarray, y_true: np.ndarray):

    class_weights = 1 / np.array([(y_true == 0).sum(), (y_true == 1).sum()])

    return log_loss(y_true, y_pred, sample_weight=class_weights[y_true], eps=1e-15)

def precision_recall(y_pred: np.ndarray, y_true: np.ndarray):

    mat = confusion_matrix(y_true, (y_pred > .5).astype(int))
    _tn, fp, fn, tp = mat.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    wrongs = fp + fn
    return precision, recall, wrongs

def compare_with_color(precision: float, recall: float, reverse: bool = False):
    def get_color(my, other):
        if my == 1. - float(reverse):
            return 'green'
        if reverse:
            return 'green' if my < other else 'red'
        return 'green' if my > other else 'red'
    precision_str = colored(f'{precision:.3f}', get_color(precision, recall))
    recall_str = colored(f'{recall:.3f}', get_color(recall, precision))
    return precision_str, recall_str
