from __future__ import annotations
from typing import overload
from typing_extensions import Self
# import os
# import pickle

# from .params import profiles
from .xgb_classifier import ICRXGBClassifier
from .lgb_classifier import ICRLGBClassifier
from .tabpfn_classifier import ICRTabPFNClassifier
# import numpy as np


class ICRClassifier:

    def __init__(
            self,
            profile: str,
            seed: int,
            *,
            cat_col_index: int,
            class_labels,
            tab_config: dict,
            # **kwargs,
        ):

        if 'xgb' in profile:
            self.classifier = ICRXGBClassifier(
                class_labels=class_labels,
                seed=seed,
                profile=profile,
            )
        
        elif 'tab' in profile:
            self.classifier = ICRTabPFNClassifier(
                profile,
                seed=seed,
                config=ICRTabPFNClassifier.Config(**tab_config),
            )
        else:
            assert 'lgb' in profile
            self.classifier = ICRLGBClassifier(
                seed=seed,
                cat_col_index=cat_col_index,
                profile=profile,
            )

        return
    
    @property
    def profile(self):
        return self.classifier.profile
    # @staticmethod
    # def _get_scale_pos_weight(labels: np.ndarray):
    #     # tem = (labels != 0).sum() / (labels == 0).sum()
    #     return 1.

    @overload
    def fit(self, x_train, y_train, x_valid, y_valid) -> Self:...

    def fit(self, x, y, x_valid = None, y_valid = None):
        self.classifier.fit(x, y, x_valid, y_valid)
        return self

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

    # def save(self, save_dir: str, name: str):
    #     with open(os.path.join(save_dir, f'{name}.pkl'), mode='wb') as fout:
    #         pickle.dump(self, fout)
    #     return
        
    # @classmethod
    # def load_classifer(cls, load_dir: str, name: str) -> ICRClassifier:
    #     with open(os.path.join(load_dir, name), mode='rb') as fin:
    #         classifier = pickle.load(fin)
    #     return classifier
