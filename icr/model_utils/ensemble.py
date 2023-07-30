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
    
    @classmethod
    def __get_params(cls, kwargs, model_name):
        for k, args in kwargs.items():
            if k in model_name:
                assert isinstance(args, dict)
                return args
        return {}

    def fit(self, x, y, **kwargs: dict):

        for name, model in self.models.items():

            params = self.__get_params(kwargs, name)
            model.fit(x, y, **params)
        return self
    
    def predict_proba(self, x, **kwargs: dict):

        # predictions = np.zeros(len(x), dtype=np.float64)
        predictions = None
        for name, model in self.models.items():
            params = self.__get_params(kwargs, name)
            res = model.predict_proba(x, **params)
            if predictions is None:
                predictions = np.zeros_like(res, dtype=np.float64)
            predictions += res
        
        return predictions / len(self.models)
