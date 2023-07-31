from __future__ import annotations
import os
from typing import overload, Union
import pickle

from lightgbm import LGBMClassifier
from .params import profiles
from .base import get_sample_weights
# import numpy as np


class ICRLGBClassifier:

    def __init__(self, cat_col_index: Union[int, None], seed: int, profile: str,**kwargs):

        assert profile in profiles
        assert 'lgb' in profile

        self.classifier = ICRLGBClassifier._get_lgb_classifier(
            profiles[profile],
            seed,
        )
        self.seed = seed
        self.kwargs = kwargs
        self.profile = profile
        self.cat_col_index = cat_col_index
        return
    
    # @staticmethod
    # def _get_scale_pos_weight(labels: np.ndarray):
    #     # tem = (labels != 0).sum() / (labels == 0).sum()
    #     return 1.

    @staticmethod
    def _get_lgb_classifier(params: dict, seed):
        return LGBMClassifier(
            **params,
            random_state = seed,
            verbose = -1,
            early_stopping_round = 999,
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
            eval_sample_weight = [get_sample_weights(y_valid)],
            categorical_feature=[self.cat_col_index] if self.cat_col_index is not None else [],
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
    def load_classifer(cls, load_dir: str, name: str) -> ICRLGBClassifier:
        with open(os.path.join(load_dir, name), mode='rb') as fin:
            classifier = pickle.load(fin)
        return classifier
