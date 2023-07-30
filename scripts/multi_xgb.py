#pylint: disable=all
import os
import warnings
import typing

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from pydantic import model_validator
from termcolor import cprint, colored

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.tools import balanced_log_loss, seed_everything
from icr.xgb import ICRXGBClassfier
from icr.model_utils.ensemble import Ensemble

from sklearn.metrics import confusion_matrix
from model_utils.config import BaseConfig

class XGBConfig(BaseConfig):

    learning_rate: float = 0.1419865761603358
    max_bin: int = 824
    min_child_weight: float = 1
    # random_state: int = 811996
    reg_alpha: float = 1.6259583347890365e-07
    reg_lambda: float = 2.110691851528507e-08
    subsample: float = 0.879020578464637
    # objective: str = 'binary:logistic'
    eval_metric: str = 'mlogloss'
    colsample_bytree: float = 0.5646751146007976
    gamma: float = 7.788727238356553e-06
    max_depth: int = 3
    n_jobs: int = -1
    verbosity: int = 0
    
    @staticmethod
    def get_scale_pos_weight(class_labels: np.ndarray):
        return (class_labels == 0).sum() / (class_labels == 1).sum()

    scale_pos_weight: float



class Config(ICRDatasetConfig):
    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'

    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    # smote_strategy: float = .5
    # labels: typing.Literal['class', 'alpha'] = 'class'
    labels: typing.Literal['class', 'alpha'] = 'alpha'
    # standard_scale_enable: bool = False
    mxgb_profile: typing.Union[str, typing.Sequence[str]]


def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    # cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = np.zeros(k)
    ps = np.zeros(k)
    rs = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    train_class = train_set.class_ser
    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_class, train_class)):
        
        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')

        train_subset = train_subset.new_over_sampled_dataset()
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)


        if isinstance(config.mxgb_profile, str):
            classifier = ICRXGBClassfier(alpha_train, seed, config.mxgb_profile)
            classifier.fit(x_train, alpha_train, x_valid, alpha_valid)
        
        else:
            classifier = Ensemble(
                {
                    profile: ICRXGBClassfier(alpha_train, seed, profile)\
                        for profile in config.mxgb_profile
                },
            )
            classifier.fit(x_train, alpha_train, xgb={'x_valid': x_valid, 'y_valid': alpha_valid})

        prediction = classifier.predict_proba(x_valid)
        # res: (n_samples, 2[class0, class1])
        # prediction = prediction.astype(np.float64)
        # prediction[prediction < .2] = 0.
        # prediction[prediction > .8] = 1.
        eval_loss = balanced_log_loss(prediction, class_valid)
        mat = confusion_matrix(class_valid, (prediction > .5).astype(int))
        tn, fp, fn, tp = mat.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f'eval_loss = {eval_loss}')
        def get_color(my, other):
            if my == 1.:
                return 'green'
            return 'green' if my > other else 'red'
        precision_str = colored(f'{precision:.3f}', get_color(precision, recall))
        recall_str = colored(f'{recall:.3f}', get_color(recall, precision))

        print(f'precision: {precision_str}, recall: {recall_str}')
        cv_loss[fold] = eval_loss
        ps[fold] = precision
        rs[fold] = recall
        
    
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    return cv_loss, ps, rs


def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = Config(
        # epochs=100,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy='auto',
            sampling_strategy={0: 509, 1: 122, 2: 36, 3: 58},
            method='smote',
        ),
        epsilon_as_feature=True,
        mxgb_profile=['mxgb1', 'mxgb2', 'mxgb3'],
    )
    config.display()
    k = 5
    s = 10
    seeds = [0xAAAAAA + i for i in range(s)]
    cv_loss = np.zeros((s, k))
    ps = np.zeros((s, k))
    rs = np.zeros((s, k))

    for idx, seed in enumerate(seeds):
        cv_loss[idx, :], ps[idx, :], rs[idx, :] = cross_validation(k, config, seed)


    print('-' * 60)
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    print('preciions:')
    print(ps.tolist())
    print('recalls:')
    print(rs.tolist())



if __name__ == '__main__':
    main()
