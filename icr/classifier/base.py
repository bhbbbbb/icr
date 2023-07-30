import numpy as np

def _get_class_weights(y: np.ndarray):
    negative_count = (y == 0).sum()
    return len(y) / np.array([negative_count, len(y) - negative_count])

def get_sample_weights(y: np.ndarray):
    class_weights = _get_class_weights(y)
    return class_weights[(y != 0).astype(int)]
