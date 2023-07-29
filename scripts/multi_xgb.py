#pylint: disable=all
import os
import warnings
import typing

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.tools import balanced_log_loss, seed_everything

from xgboost import XGBClassifier
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
    smote_strategy: dict = {0: 509, 1: 122, 2: 36, 3: 58}
    labels: typing.Literal['class', 'alpha'] = 'alpha'
    # standard_scale_enable: bool = False
    xgb_params: dict = {}


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
        x_train, y_train = train_subset[:]
        x_valid, y_valid = valid_subset[:]
        class_train = (y_train != 0).astype(int)
        class_valid = (y_valid != 0).astype(int)

        class_weights_train = len(y_train) / np.array([(y_train == 0).sum(), (y_train != 0).sum()])
        class_weights_valid = len(y_valid) / np.array([(y_valid == 0).sum(), (y_valid != 0).sum()])

        xgb_config = XGBConfig(
            **{
                **config.xgb_params,
                'scale_pos_weight': XGBConfig.get_scale_pos_weight(class_train)
            }
        )
        classifier = XGBClassifier(**{**xgb_config.model_dump(), 'random_state': seed})
        # classifier = XGBClassifier(random_state = seed)
        # classifier = RandomForestClassifier(
        #     max_depth=3,
        #     verbose=1,
        #     criterion='log_loss',
        #     class_weight=class_weights_train,
        # )
        classifier.fit(
            x_train, y_train,
            sample_weight=class_weights_train[class_train],
            eval_set = [(x_valid, y_valid)],
            sample_weight_eval_set = [class_weights_valid[class_valid]],
            # verbose = True,
            verbose = False
        )

        res = classifier.predict_proba(x_valid)
        # res: (n_samples, 2[class0, class1])
        prediction = res[:, 0].squeeze()
        prediction = 1 - prediction
        # prediction = prediction.astype(np.float64)
        # prediction[prediction < .2] = 0.
        # prediction[prediction > .8] = 1.
        eval_loss = balanced_log_loss(prediction, class_valid)
        mat = confusion_matrix(class_valid, (prediction > .5).astype(int))
        tn, fp, fn, tp = mat.ravel()
        precsion = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f'eval_loss = {eval_loss}')
        print(f'precision: {precsion}, recall: {recall}')
        cv_loss[fold] = eval_loss
        ps[fold] = precsion
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
            sampling_strategy='auto',
            method='random',
        ),
        epsilon_as_feature=True,
    )
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
    print(ps.tolist())
    print(rs.tolist())



if __name__ == '__main__':
    main()
