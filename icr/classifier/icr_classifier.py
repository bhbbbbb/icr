from __future__ import annotations
import os
from typing import overload
import pickle

# from .params import profiles
from .xgb_classifier import ICRXGBClassfier
from .lgb_classifier import ICRLGBClassifier
# import numpy as np


class ICRClassifier:

    def __init__(
            self,
            profile: str,
            cat_col_index: int,
            class_labels,
            seed: int,
            # **kwargs,
        ):

        if 'xgb' in profile:
            self.classifier = ICRXGBClassfier(
                class_labels=class_labels,
                seed=seed,
                profile=profile,
            )
        
        else:
            assert 'lgb' in profile
            self.classifier = ICRLGBClassifier(
                seed=seed,
                cat_col_index=cat_col_index,
                profile=profile,
            )

        return
    
    # @staticmethod
    # def _get_scale_pos_weight(labels: np.ndarray):
    #     # tem = (labels != 0).sum() / (labels == 0).sum()
    #     return 1.

    @overload
    def fit(self, x_train, y_train, x_valid, y_valid):...

    def fit(self, x, y, x_valid = None, y_valid = None):
        return self.classifier.fit(x, y, x_valid, y_valid)

    def predict_proba(self, x, no_reshape: bool = False, **_kwargs):
        """_summary_

        Args:
            x (_type_): _description_
            no_reshape (bool): pass True to return original output of predcit_prob

        Returns:
            np.ndarry (n_samples, )
        """
        return self.classifier.predict_proba(x, no_reshape=no_reshape, **_kwargs)
    
    def set_params(self, **params):
        return self.classifier.set_params(**params)

    def save(self, save_dir: str, name: str):
        with open(os.path.join(save_dir, f'{name}.pkl'), mode='wb') as fout:
            pickle.dump(self, fout)
        return
        
    @classmethod
    def load_classifer(cls, load_dir: str, name: str) -> ICRClassifier:
        with open(os.path.join(load_dir, name), mode='rb') as fin:
            classifier = pickle.load(fin)
        return classifier