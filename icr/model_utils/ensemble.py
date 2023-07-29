from typing import Protocol, Sequence, Dict
import numpy as np

class _Model(Protocol):

    def fit(self, x, y) -> None:...

    def predict_prob(self, x) -> np.ndarray:...


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
        return
    
    def predict_prob(self, x):

        predictions = np.zeros(len(x), dtype=np.float64)
        for model in self.models.values():
            res = model.predict_prob(x)
            prediction = res[:, 0].squeeze()
            prediction = 1 - prediction
            predictions += prediction
        
        return predictions / len(self.models)
 
