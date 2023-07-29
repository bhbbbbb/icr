from typing import Protocol, Sequence, Dict
import numpy as np

class _Model(Protocol):

    def fit(self, x, y) -> None:...

    def predict_proba(self, x) -> np.ndarray:...


class Ensemble:

    models: Dict[str, _Model]

    def __init__(self, models: Dict[str, _Model]):
        self.models = models
        return
    
    def fit(self, x, y, **kwargs: dict):

        for name, model in self.models.items():

            def get_params():
                for k, args in kwargs.items():
                    if k in name:
                        return args
                return {}
            params = get_params()
            model.fit(x, y, **params)
        return self
    
    def predict_proba(self, x):

        predictions = np.zeros(len(x), dtype=np.float64)
        for model in self.models.values():
            res = model.predict_proba(x)
            predictions += res
        
        return predictions / len(self.models)
