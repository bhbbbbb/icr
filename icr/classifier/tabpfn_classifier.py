from __future__ import annotations
import os
import pickle
from typing import TypedDict

from tabpfn import TabPFNClassifier
# from model_utils.base import BaseConfig
from .params import profiles
from imblearn.over_sampling import RandomOverSampler
import numpy as np


class ICRTabPFNClassifier:

    class Config(TypedDict):
        device: str
        base_path: str

    def __init__(self, profile: str, seed: int, config: Config):

        assert profile in profiles
        assert 'tab' in profile

        self.classifier = TabPFNClassifier(
            N_ensemble_configurations=profiles[profile],
            seed=seed,
            **config
        )
        self.seed = seed
        self.profile = profile
        return
    
    def fit(self, x_train, y_train, _x_valid, _y_valid):

        def get_counts():
            labels = np.unique(y_train)
            for label in labels:
                if label != 0:
                    yield label, (y_train == label).sum() * 2
                else:
                    yield label, (y_train == label).sum()

        sampling_strategy = dict(get_counts())
        if sampling_strategy[1] > sampling_strategy[0]:
            sampling_strategy[1] = sampling_strategy[0]

        over_sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.seed,
        )
        x_train, y_train = over_sampler.fit_resample(x_train, y_train)
        return self.classifier.fit(x_train, y_train)

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
    def load_classifer(cls, load_dir: str, name: str) -> ICRTabPFNClassifier:
        with open(os.path.join(load_dir, name), mode='rb') as fin:
            classifier = pickle.load(fin)
        return classifier
