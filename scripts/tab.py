#pylint: disable=all
import os
import warnings
import typing

from sklearn.model_selection import StratifiedKFold
import numpy as np

from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.tools import balanced_log_loss, seed_everything, precision_recall, color_precision_recall

from tabpfn import TabPFNClassifier
from sklearn.metrics import confusion_matrix
from model_utils.config import BaseConfig

dataset_dir = os.path.realpath('icr-identify-age-related-conditions')

class Config(ICRDatasetConfig):
    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'
    greeks_csv_path: str = os.path.join(dataset_dir, 'greeks.csv')

    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    standard_scale_enable: bool = False

def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    # cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    ps = np.zeros(k)
    rs = np.zeros(k)

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(*(train_set[:]))):
        
        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')

        train_subset = train_subset.new_over_sampled_dataset()
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)

        classifier = TabPFNClassifier(
            device='cuda:0', N_ensemble_configurations=64, seed=seed,
            base_path=dataset_dir,
        )

        classifier.fit(x_train, alpha_train)

        res = classifier.predict_proba(x_valid)
        predictions = (1 - res[:, 0]).squeeze()

        eval_loss = balanced_log_loss(predictions, class_valid)
        print(f'eval_loss = {eval_loss}')
        precision, recall = precision_recall(predictions, class_valid)
        precision_str, recall_str = color_precision_recall(precision, recall)
        print(f'precision: {precision_str}, recall: {recall_str}')
        cv_loss[fold] = eval_loss
        ps[fold] = precision
        rs[fold] = recall
    
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    return cv_loss


def main():

    config = Config(
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        over_sampling_config=Config.OverSamplingConfig(
            sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47},
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
    )
    
    k = 5
    s = 10
    seeds = [0xAAAAAA + i for i in range(s)]
    cv_loss = np.zeros((len(seeds), k))
    for idx, seed in enumerate(seeds):
        cv_loss[idx, :] = cross_validation(k, config, seed)
        return
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')


if __name__ == '__main__':
    main()
