#pylint: disable=all
import os
import random
import torch
import warnings
import typing

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

from model_utils import formatted_now
from model_utils.config import BaseConfig
from pydantic import field_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.models import ICRDNNModel, ICRDNNConfig, CNNModelConfig, CNNModel
from icr.model_utils.icr_model_utils import ICRModelUtils
from icr.model_utils.icr_model_utils_config import ICRModelUtilsConfig

SEEDS = [0xAAAAAAA + i for i in range(5)]

class Config(ICRModelUtilsConfig, ICRDatasetConfig):
    loss_class_weights: typing.Tuple[float, float] = [1., 1.]
    epochs_per_checkpoint: int = 0

    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'
    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    early_stopping_rounds: int = 20
    save_n_best: int = 1

    icr_model_config: typing.Union[CNNModelConfig, ICRDNNConfig]


def cross_validation(
        log_title: str,
        k: int,
        ICRModel: type,
        config: Config,
        seeds: typing.List[int],
    ):

    log_dir = os.path.join(config.log_dir, f'{log_title}-{formatted_now()}/')

    all_loss = np.zeros(k * len(seeds))
    for idx, seed in enumerate(seeds):
        cv_loss = _cross_validation(log_dir, k, ICRModel, config, seed)
        start_idx = idx * k
        all_loss[start_idx:(start_idx + k)] = cv_loss
    
    print(all_loss)
    print(f'all_loss- mean-{all_loss.mean()}, std-{all_loss.std()}')
    with open(os.path.join(log_dir, 'all_loss.txt'), 'w', encoding='utf8') as fout:
        fout.write(f'{all_loss.tolist()}\n')
        fout.write(f'{all_loss.mean()}, {all_loss.std()}')
    return all_loss


def _cross_validation(
        log_dir: str,
        k: int,
        ICRModel: type,
        config: Config,
        seed: int = 0xAAAA,
    ):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    cv_log_dir = os.path.join(log_dir, f'seed-{seed}')
    # cv_loss = 0.
    cv_loss: np.ndarray = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(*(train_set[:]))):
        

        train_indices: typing.List[int]
        train_indices: typing.List[int]
        # print(f'\n --------------- Fold: {fold} ------------------')
        train_loader = train_set.make_subset(train_indices, 'train').dataloader
        valid_loader = train_set.make_subset(valid_indices, 'valid').dataloader
        config = Config(
            **{
                **config.model_dump(),
                'log_dir': os.path.join(cv_log_dir, f'fold-{fold}'),
                'steps_per_epoch': len(train_loader),
            }
        )
        config.display()

        model = ICRModel(config.icr_model_config)
        model_utils: ICRModelUtils = ICRModelUtils.start_new_training(model, config)

        history = model_utils.train(config.epochs, train_loader, valid_loader)

        best_valid_loss = history.get_best_criterion()
        # cv_loss += best_valid_loss.value
        cv_loss[fold] = best_valid_loss.value

        # model_utils.plot_history()
    
    # cv_loss /= k

    print(cv_loss)
    print(f'cv_loss = {cv_loss.mean():.6f}, {cv_loss.std(): .6f}')
    with open(os.path.join(cv_log_dir, 'cv_loss.txt'), 'w', encoding='utf8') as fout:
        fout.write(f'{cv_loss.tolist()}\n')
        fout.write(f'{cv_loss.mean()}, {cv_loss.std()}')
    return cv_loss


def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = Config(
        epochs=300,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        progress_bar=False,
        icr_model_config=ICRDNNConfig(),
        loss_class_weights=(1., 1.),
        over_sampling_strategy=1.,
        save_n_best=0,
    )
    cross_validation('dnn-smote', 10, ICRDNNModel, config, seeds=SEEDS)

    # dir_path = 'log/cv-20230727T00-07-03/fold-0/20230727T00-07-03'
    # model = ICRModel(config)
    # utils = ICRModelUtils.load_last_checkpoint(model, dir_path)
    # utils.plot_history()
    # config = utils.config
    # config.display()
    return

if __name__ == '__main__':
    main()
