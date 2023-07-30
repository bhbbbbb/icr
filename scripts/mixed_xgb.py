#pylint: disable=all
import os
import warnings
import typing

from termcolor import cprint, colored
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd

from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.xgb import ICRXGBClassfier
from icr.model_utils.ensemble import Ensemble

from icr.tools import balanced_log_loss, seed_everything, precision_recall, color_precision_recall

from sklearn.metrics import confusion_matrix
from model_utils.config import BaseConfig


class Config(ICRDatasetConfig):
    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'

    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4

    # over_sampling_config: ICRDatasetConfig.OverSamplingConfig =\
    # smote_strategy: float = .5
    # labels: typing.Literal['class', 'alpha'] = 'class'
    # smote_strategy: dict = {0: 509, 1: 122, 2: 36, 3: 58}
    # labels: typing.Literal['class', 'alpha'] = 'alpha'
    # standard_scale_enable: bool = False
    xgb_profile: typing.Union[str, typing.Sequence[str]]
    mxgb_profile: typing.Union[str, typing.Sequence[str]]
    inner_over_sampling_config: ICRDatasetConfig.OverSamplingConfig


def inner_cv(
        train_subset: ICRDataset,
        valid_subset: ICRDataset,
        test_subset: ICRDataset,
        config: Config,
        seed,
    ):

    assert config.labels == 'alpha'
    train_subset = train_subset.new_over_sampled_dataset(config.inner_over_sampling_config)
    x_train, alpha_train = train_subset[:]
    x_valid, alpha_valid = valid_subset[:]

    if isinstance(config.mxgb_profile, str):
        mul_classifier = ICRXGBClassfier(alpha_train, seed, config.mxgb_profile)
        mul_classifier.fit(x_train, alpha_train, x_valid, alpha_valid)
    
    else:
        mul_classifier = Ensemble(
            {
                profile: ICRXGBClassfier(alpha_train, seed, profile)\
                    for profile in config.mxgb_profile
            },
        )
        mul_classifier.fit(x_train, alpha_train, mxgb={'x_valid': x_valid, 'y_valid': alpha_valid})

    mul_prediction = mul_classifier.predict_proba(
                            x_valid, no_reshape=True, mxgb={'no_reshape': True})
    
    prediction = 1 - mul_prediction[:, 0]
    class_valid = (alpha_valid != 0).astype(int)
    eval_loss = balanced_log_loss(prediction, class_valid)
    print(f'inner_eval_loss = {eval_loss}')
    precision, recall = precision_recall(prediction, class_valid)
    precision_str, recall_str = color_precision_recall(precision, recall)
    print(f'inner_precision: {precision_str}, recall: {recall_str}')

    x_test, alpha_test = test_subset[:]
    test_prediction = mul_classifier.predict_proba(
                                    x_test, no_reshape=True, mxgb={'no_reshape': True})

    return mul_prediction, test_prediction
    


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
        train_class = train_subset.class_ser

        oof_predictions = np.zeros((len(train_subset), 4), dtype=np.float64)
        valid_alpha_prediction = np.zeros((len(valid_subset), 4), dtype=np.float64)
        for inner_fold, (inner_train_indices, inner_valid_indicies) in\
            enumerate(spliter.split(train_class, train_class)):
            oof_prediction, test_prediction = inner_cv(
                train_subset.make_subset(inner_train_indices, 'train'),
                train_subset.make_subset(inner_valid_indicies, 'valid'),
                valid_subset,
                config,
                seed,
            )
            oof_predictions[inner_valid_indicies] = oof_prediction
            valid_alpha_prediction += test_prediction
        
        valid_alpha_prediction /= k
        
        train_predicted_alpha_df = pd.DataFrame(oof_predictions)
        valid_predicted_alpha_df = pd.DataFrame(valid_alpha_prediction)


        train_subset.df = pd.concat(
            [train_subset.df.reset_index(drop=True), train_predicted_alpha_df], axis=1)
        valid_subset.df = pd.concat(
            [valid_subset.df.reset_index(drop=True), valid_predicted_alpha_df], axis=1)

        train_subset = train_subset.new_over_sampled_dataset()
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)

        if isinstance(config.xgb_profile, str):
            classifier = ICRXGBClassfier(class_train, seed, config.xgb_profile)
            classifier.fit(x_train, class_train, x_valid, class_valid)
        
        else:
            classifier = Ensemble(
                {
                    profile: ICRXGBClassfier(class_train, seed, profile)\
                        for profile in config.xgb_profile
                },
            )
            classifier.fit(x_train, class_train, xgb={'x_valid': x_valid, 'y_valid': class_valid})
        

        prediction = classifier.predict_proba(x_valid)

        # bin_prediction_sampled = (bin_prediction > .5).astype(int)

        # prediction = prediction.astype(np.float64)
        # prediction[prediction < .2] = 0.
        # prediction[prediction > .8] = 1.
        eval_loss = balanced_log_loss(prediction, class_valid)
        print(f'eval_loss = {eval_loss}')
        precision, recall = precision_recall(prediction, class_valid)
        precision_str, recall_str = color_precision_recall(precision, recall)
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
        # xgb_profile='xgb3',
        # mxgb_profile='mxgb3'
        xgb_profile=['xgb1', 'xgb2', 'xgb3'],
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
    print('precisions')
    print(ps.tolist())
    print('recalls')
    print(rs.tolist())



if __name__ == '__main__':
    main()
