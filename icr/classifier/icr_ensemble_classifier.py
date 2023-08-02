from __future__ import annotations
from typing import Sequence
# import os
# import pickle

# from .params import profiles
# from .xgb_classifier import ICRXGBClassifier
# from .lgb_classifier import ICRLGBClassifier
# from .tabpfn_classifier import ICRTabPFNClassifier
from ..tools.ensemble import ensemble_proba
from .icr_classifier import ICRClassifier
# import numpy as np


class ICREnsembleClassifier:

    def __init__(
            self,
            profiles: Sequence[str],
            seed: int,
            *,
            cat_col_index: int,
            class_labels,
            tab_config: dict,
        ):

        self.classifiers = [
            ICRClassifier(
                profile,
                seed,
                cat_col_index=cat_col_index,
                class_labels=class_labels,
                tab_config=tab_config,
            )\
            for profile in profiles
        ]
        return
    
    # @property
    # def profile(self):
    #     return self.classifier.profile

    def fit(
        self,
        x_train,
        *,
        alpha_train = None,
        x_valid = None,
        alpha_valid = None,
    ):
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)
        for classifier in self.classifiers:
            y_train = alpha_train if 'm' in classifier.profile else class_train
            y_valid = alpha_valid if 'm' in classifier.profile else class_valid
            assert y_train is not None
            classifier.fit(x_train, y_train, x_valid, y_valid)
        return self

    def predict_proba(self, x, no_reshape: bool = False, **_kwargs):
        """_summary_

        Args:
            x (_type_): _description_
            no_reshape (bool): pass True to return original output of predcit_prob

        Returns:
            np.ndarry (n_samples, )
        """
        return ensemble_proba(
            classifier.predict_proba(x, no_reshape=no_reshape, **_kwargs)
                for classifier in self.classifiers
        )
    
    # def save(self, save_dir: str, name: str):
    #     with open(os.path.join(save_dir, f'{name}.pkl'), mode='wb') as fout:
    #         pickle.dump(self, fout)
    #     return
        
    # @classmethod
    # def load_classifer(cls, load_dir: str, name: str) -> ICRClassifier:
    #     with open(os.path.join(load_dir, name), mode='rb') as fin:
    #         classifier = pickle.load(fin)
    #     return classifier
