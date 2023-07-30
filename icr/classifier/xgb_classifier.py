from __future__ import annotations
import os
import pickle
from typing import overload
import numpy as np
from xgboost import XGBClassifier
from .params import profiles
from .base import get_sample_weights


class ICRXGBClassfier:

    def __init__(self, class_labels: np.ndarray, seed: int, profile: str, **kwargs):

        assert profile in profiles
        assert 'xgb' in profile

        self.classifier = ICRXGBClassfier._get_xgb_classifier(
            profiles[profile],
            class_labels,
            seed,
        )
        self.seed = seed
        self.kwargs = kwargs
        self.profile = profile
        return
    

    @staticmethod
    def _get_scale_pos_weight(_labels: np.ndarray):
        # tem = (labels != 0).sum() / (labels == 0).sum()
        return 1.

    @staticmethod
    def _get_xgb_classifier(params: dict, class_labels: np.ndarray, seed):
        return XGBClassifier(
            **params,
            scale_pos_weight = ICRXGBClassfier._get_scale_pos_weight(class_labels),
            early_stopping_rounds = 999,
            random_state = seed,
        )
    
    @overload
    def fit(self, x_train, y_train, x_valid, y_valid):...

    def fit(self, x, y, x_valid = None, y_valid = None):
        if x_valid is None:
            x_valid = self.kwargs.get('x_valid', None)
        if y_valid is None:
            y_valid = self.kwargs.get('y_valid', None)

        assert x_valid is not None and y_valid is not None
        self._fit(x, y, x_valid, y_valid)
        return

    def _fit(self, x_train, y_train, x_valid, y_valid):
        self.classifier.fit(
            x_train, y_train,
            sample_weight=get_sample_weights(y_train),
            eval_set = [(x_valid, y_valid)],
            sample_weight_eval_set = [get_sample_weights(y_valid)],
            # verbose = True,
            verbose = False,
        )
        return self

    def predict_proba(self, x, no_reshape: bool = False, **_kwargs):
        """_summary_

        Args:
            x (_type_): _description_
            no_reshape (bool): pass True to return original output of predcit_prob

        Returns:
            np.ndarry (n_samples, )
        """
        res = self.classifier.predict_proba(x)

        return res if no_reshape else (1. - res[:, 0]).squeeze()
    
    def set_params(self, **params):
        return self.classifier.set_params(**params)


    def save(self, save_dir: str, name: str):
        with open(os.path.join(save_dir, f'{name}.pkl'), mode='wb') as fout:
            pickle.dump(self, fout)
        return
        
    @classmethod
    def load_classifer(cls, load_dir: str, name: str) -> ICRXGBClassfier:
        with open(os.path.join(load_dir, name), mode='rb') as fin:
            classifier = pickle.load(fin)
        return classifier
