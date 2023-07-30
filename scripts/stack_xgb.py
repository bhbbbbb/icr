#pylint: disable=all
import os
import warnings
import typing

from termcolor import cprint, colored
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.xgb import ICRXGBClassfier
from icr.model_utils.ensemble import Ensemble

from icr.tools import balanced_log_loss, seed_everything, precision_recall, color_precision_recall

from model_utils.config import BaseConfig


dataset_dir = os.path.realpath('icr-identify-age-related-conditions')

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
    inner_xgb_profiles: typing.Sequence[str]
    # mxgb_profile: typing.Union[str, typing.Sequence[str]]
    inner_over_sampling_config: ICRDatasetConfig.OverSamplingConfig
    passthrough: bool


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
    x_test, _alpha_test = test_subset[:]
    class_train = (alpha_train != 0).astype(int)
    class_valid = (alpha_valid != 0).astype(int)

    oof_predictions = []
    test_predictions = []
    models: typing.List[ICRXGBClassfier] = []
    for profile in config.inner_xgb_profiles:
        classifier = ICRXGBClassfier(alpha_train, seed, profile)
        y_train = alpha_train if 'mxgb' in profile else class_train
        y_valid = alpha_valid if 'mxgb' in profile else class_valid
        classifier.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)

        prediction = classifier.predict_proba(x_valid, no_reshape=('mxgb' in profile))
        test_prediction = classifier.predict_proba(x_test, no_reshape=('mxgb' in profile))
        # bxgb: (n_samples, )
        # mxgb: (n_samples, 4)

        models.append(classifier)
        oof_predictions.append(prediction if 'mxgb' in profile else prediction.reshape(-1, 1))
        test_predictions.append(test_prediction if 'mxgb' in profile else test_prediction.reshape(-1, 1))

    oof_predictions = np.concatenate(oof_predictions, axis=1)
    test_predictions = np.concatenate(test_predictions, axis=1)
    return models, oof_predictions, test_predictions


def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    # cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = np.zeros(k)
    best_loss = 100000.
    best_modelss = None
    ps = np.zeros(k)
    rs = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_class = train_set.class_ser

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_class, train_class)):

        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')
        train_class = train_subset.class_ser

        inner_dim = sum(4 if 'mxgb' in profile else 1 for profile in config.inner_xgb_profiles)
        modelss: typing.List[typing.List[ICRXGBClassfier]] = []
        oof_predictionss = np.zeros((len(train_subset), inner_dim), dtype=np.float64)
        valid_predictions = np.zeros((len(valid_subset), inner_dim), dtype=np.float64)
        for inner_fold, (inner_train_indices, inner_valid_indicies) in\
            enumerate(spliter.split(train_class, train_class)):
            models, oof_predictions, test_predictions = inner_cv(
                train_subset.make_subset(inner_train_indices, 'train'),
                train_subset.make_subset(inner_valid_indicies, 'valid'),
                valid_subset,
                config,
                seed,
            )
            modelss.append(models)
            oof_predictionss[inner_valid_indicies] = oof_predictions
            valid_predictions += test_predictions
        
        valid_predictions /= k
        
        train_predicted_alpha_df = pd.DataFrame(oof_predictionss)
        valid_predicted_alpha_df = pd.DataFrame(valid_predictions)


        if config.passthrough:
            train_subset.df = pd.concat(
                [train_subset.df.reset_index(drop=True), train_predicted_alpha_df], axis=1)
            valid_subset.df = pd.concat(
                [valid_subset.df.reset_index(drop=True), valid_predicted_alpha_df], axis=1)
            train_subset = train_subset.new_over_sampled_dataset()
        else:
            train_subset.df = train_predicted_alpha_df
            valid_subset.df = valid_predicted_alpha_df
        
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
        # prediction = prediction[:, 1]
        # prediction = x_valid.mean(axis=1)

        # bin_prediction_sampled = (bin_prediction > .5).astype(int)

        # prediction = prediction.astype(np.float64)
        # prediction[prediction < .2] = 0.
        # prediction[prediction > .8] = 1.
        eval_loss = balanced_log_loss(prediction, class_valid)
        print(f'eval_loss = {eval_loss}')
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_modelss = modelss
        precision, recall = precision_recall(prediction, class_valid)
        precision_str, recall_str = color_precision_recall(precision, recall)
        print(f'precision: {precision_str}, recall: {recall_str}')
        cv_loss[fold] = eval_loss
        ps[fold] = precision
        rs[fold] = recall

        if fold == k - 1:
            model_save_dir = os.path.join(dataset_dir, 'models')
            for f, models in enumerate(best_modelss):
                save_dir = os.path.join(model_save_dir, f'fold-{f}')
                os.makedirs(save_dir, exist_ok=True)
                for model in models:
                    model.save(save_dir, model.profile)

        
    
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    return cv_loss, ps, rs


def main():
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
        # passthrough=False,
        passthrough=True,
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
