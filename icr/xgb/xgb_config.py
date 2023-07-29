import numpy as np
from model_utils.config import BaseConfig

class XGBConfig(BaseConfig):

    # n_estimators: int = 100
    # max_depth: int = 5
    # learning_rate: float = 0.41
    # objective: str = 'binary:logistic'
    # eval_metric: str = 'logloss'
    # subsample: float = .24
    early_stopping_rounds: int = 999
    learning_rate: float = 0.1419865761603358
    max_bin: int = 824
    min_child_weight: float = 1
    # random_state: int = 811996
    reg_alpha: float = 1.6259583347890365e-07
    reg_lambda: float = 2.110691851528507e-08
    subsample: float = 0.879020578464637
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    colsample_bytree: float = 0.5646751146007976
    gamma: float = 7.788727238356553e-06
    max_depth: int = 3
    n_jobs: int = -1
    verbosity: int = 0
    
    @staticmethod
    def get_scale_pos_weight(labels: np.ndarray):
        return (labels == 0).sum() / (labels != 0).sum()

    scale_pos_weight: float

