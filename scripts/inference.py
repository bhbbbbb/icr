#pylint: disable=all
import os
import warnings
import typing
from typing import Sequence as Seq

from termcolor import cprint, colored
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd

from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.classifier import ICRXGBClassfier
from icr.model_utils.ensemble import Ensemble

from icr.tools import balanced_log_loss, seed_everything, precision_recall, color_precision_recall

from sklearn.metrics import confusion_matrix
from model_utils.config import BaseConfig


class Config(ICRDatasetConfig):
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'

    batch_size_train: int = 64
    batch_size_eval: int = 128
    # persistent_workers: bool = False
    # num_workers: int = 0
    # pin_memory: bool = True
    # learning_rate: float = 1e-4

    # over_sampling_config: ICRDatasetConfig.OverSamplingConfig =\
    # smote_strategy: float = .5
    # labels: typing.Literal['class', 'alpha'] = 'class'
    # smote_strategy: dict = {0: 509, 1: 122, 2: 36, 3: 58}
    # labels: typing.Literal['class', 'alpha'] = 'alpha'
    # standard_scale_enable: bool = False
    xgb_profile: typing.Union[str, typing.Sequence[str]]
    inner_xgb_profiles: typing.Sequence[str]
    # mxgb_profile: typing.Union[str, typing.Sequence[str]]
    inner_over_sampling_config: ICRDatasetConfig.OverSamplingConfig
    passthrough: bool


def inference(config: ICRDatasetConfig, models: Seq[ICRXGBClassfier]):

    infer_dataset = ICRDataset('infer', config)
    x = infer_dataset[:]

    predictions = np.zeros(len(infer_dataset), dtype=np.float64)
    for model in models:
        prediction = model.predict_proba(x)
        predictions += prediction

    predictions /= len(models)

    return predictions

def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = Config(
        # epochs=100,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy=.5,
            sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47},
            method='smote',
        ),
        inner_over_sampling_config=Config.OverSamplingConfig(
            sampling_strategy={0: 338, 1: 79, 2: 24, 3: 38},
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
        xgb_profile='xgb3',
        # inner_xgb_profiles=['mxgb3', 'xgb3'],
        inner_xgb_profiles=['xgb1', 'xgb2', 'xgb3'],
        # xgb_profile=['xgb1', 'xgb2', 'xgb3'],
        passthrough=False,
    )
    config.display()

    models = list(load_model(os.path.join(dataset_dir, 'models')))
    predictions = inference(config, models)

def load_model(model_dir: str):
    for fold_dir in os.listdir(model_dir):
        fold_dir = os.path.join(model_dir, fold_dir)
        yield from [
            ICRXGBClassfier.load_classifer(fold_dir, name) for name in os.listdir(fold_dir)
        ]


if __name__ == '__main__':
    main()
