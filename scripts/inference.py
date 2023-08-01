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
# from icr.xgb import ICRXGBClassfier
from icr.classifier import ICRClassifier
from icr.model_utils.ensemble import Ensemble

from icr.tools import balanced_log_loss, seed_everything, precision_recall, compare_with_color

from sklearn.metrics import confusion_matrix
from model_utils.config import BaseConfig


class Config(ICRDatasetConfig):
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'

    batch_size_train: int = 64
    batch_size_eval: int = 128
    # persistent_workers: bool = False
    stacking_profiles: typing.Sequence[str]
    inner_profiles: typing.Sequence[str]
    inner_over_sampling_config: ICRDatasetConfig.OverSamplingConfig
    passthrough: bool
    model_save_dir: str
    tabpfn_config: dict
    prediction_analysis: bool = False

def inference(config: ICRDatasetConfig, models: Seq[ICRClassifier]):

    infer_dataset = ICRDataset('infer', config)
    x = infer_dataset[:]

    predictions = np.zeros(len(infer_dataset), dtype=np.float64)
    for model in models:
        prediction = model.predict_proba(x)
        predictions += prediction

    predictions /= len(models)

    return predictions

def load_stack_models(models_dir: str, one_seed: bool):

    def list_dirs(a_dir: str):
        for name in os.listdir(a_dir):
            if os.path.isdir(os.path.join(a_dir, name)):
                yield os.path.join(a_dir, name)

    def load_fold_models(fold_dir: str):
        for name in os.listdir(fold_dir):
            if os.path.isfile(os.path.join(fold_dir, name)):
                yield ICRClassifier.load_classifer(fold_dir, name)

    def gen_inner_modelss(fold_dir: str):
        for inner_fold_dir in list_dirs(fold_dir):
            yield load_fold_models(inner_fold_dir)

    seed_dirs = [models_dir] if one_seed else list_dirs(models_dir)
    for seed_dir in seed_dirs:
        for fold_dir in list_dirs(seed_dir):
            stack_models = load_fold_models(fold_dir)

            inner_modelss = gen_inner_modelss(fold_dir)

            yield inner_modelss, stack_models


def _inner_predict_probas(inner_models: Seq[ICRClassifier], x_test: np.ndarray):

    test_predictions = []
    for model in inner_models:
        pred = model.predict_proba(x_test, no_reshape=('m' in model.profile))
        test_predictions.append(
            pred if 'm' in model.profile else pred.reshape(-1, 1))
    return np.concatenate(test_predictions, axis=1)

def _ensemble_proba(predictions: Seq[np.ndarray]) -> np.ndarray:

    res = None
    i = 0
    for pred in predictions:
        if res is None:
            res = np.zeros_like(pred, dtype=np.float64)
        res += pred
        i += 1
    return res / i
        
def _inner_ensemble_probas(inner_modelss: Seq[Seq[ICRClassifier]], x_test):
    return _ensemble_proba(
        _inner_predict_probas(inner_models, x_test) for inner_models in inner_modelss
    )

def _ensemble_predict_proba(models: Seq[ICRClassifier], x_test: np.ndarray):
    return _ensemble_proba(
        model.predict_proba(x_test) for model in models
    )


def stack_inference(config: ICRDatasetConfig, models_dir: str, one_seed: bool):

    from tqdm import tqdm
    def run_seeds_outer_folds():
        stack_models = list(load_stack_models(models_dir, one_seed))
        for inner_modelss, stack_models in tqdm(stack_models):
            # per seed per outer-fold

            dataset = ICRDataset('infer', config)

            inner_predictions = _inner_ensemble_probas(inner_modelss, dataset[:])
            inner_predicted_df = pd.DataFrame(inner_predictions)
            dataset.test_df = pd.concat(
                [dataset.test_df.reset_index(drop=True), inner_predicted_df], axis=1)
            
            yield _ensemble_predict_proba(stack_models, dataset[:])
        
    return _ensemble_proba(run_seeds_outer_folds())

def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = Config(
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        model_save_dir=os.path.join(dataset_dir, 'models'),
        tabpfn_config={'device': 'cuda:0', 'base_path': dataset_dir},
        # tabpfn_config={'device': 'cpu', 'base_path': dataset_dir},
        over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy=.5,
            # sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47}, # k=5
            sampling_strategy={0: 459, 1: 110, 2: 33, 3: 53}, # k = 10
            method='smote',
        ),
        inner_over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy={0: 327, 1: 79, 2: 24, 3: 38}, # k = 5
            sampling_strategy={0: 414, 1: 99, 2: 30, 3: 48}, # k = 10
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
        inner_profiles=[*(f'lgb{i}' for i in range(1, 4)), *(f'xgb{i}' for i in range(1, 4)), 'tab0', 'mtab1'],
        # inner_profiles=['lgb2', 'xgb2', 'tab0', 'mtab1'],
        # inner_profiles=['lgb1'],
        # inner_profiles=[*(f'lgb{i}' for i in range(1, 5)), *(f'xgb{i}' for i in range(1, 6)), 'tab0', 'mtab1'],
        # stacking_profiles=['lgb1'],
        stacking_profiles=['lgb1', 'lgb2', 'lgb3', 'lgb4'],
        # passthrough=False,
        passthrough=True,
        prediction_analysis=True,
    )
    config.display()

    models_dir = 'D:/Documents/PROgram/ML/kaggle/icr/icr-identify-age-related-conditions/models/seed-0xaaaaab'
    # models = list(load_model(os.path.join(dataset_dir, 'models')))
    predictions = stack_inference(config, models_dir, one_seed=True)
    print(predictions)
    return

def load_model(model_dir: str):
    for fold_dir in os.listdir(model_dir):
        fold_dir = os.path.join(model_dir, fold_dir)
        yield from [
            ICRXGBClassfier.load_classifer(fold_dir, name) for name in os.listdir(fold_dir)
        ]


if __name__ == '__main__':
    main()
