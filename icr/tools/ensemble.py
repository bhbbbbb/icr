from typing import Sequence
import numpy as np

def ensemble_proba(predictions: Sequence[np.ndarray]) -> np.ndarray:
    res = None
    i = 0
    for pred in predictions:
        if res is None:
            res = np.zeros_like(pred, dtype=np.float64)
        res += pred
        i += 1
    return res / i
