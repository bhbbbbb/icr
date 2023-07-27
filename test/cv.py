#pylint: disable=all
import os
import warnings
import typing

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

from model_utils import formatted_now
from pydantic import model_validator

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.models import ICRDNNModel as ICRModel, ICRDNNConfig
from icr.model_utils.icr_model_utils import ICRModelUtils
from icr.model_utils.config import ICRModelUtilsConfig


class Config(ICRModelUtilsConfig, ICRDNNConfig, ICRDatasetConfig):
    loss_class_weights: typing.Tuple[float, float] = [1., 1.]
    save_best: bool = False
    epochs_per_checkpoint: int = 0

    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'
    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    early_stopping_threshold: int = 50

    @model_validator(mode='after')
    def set_steps_per_epoch(self):
        if self.steps_per_epoch is not None:
            assert (
                self.steps_per_epoch ==
                self.n_train_samples_per_epoch / self.batch_size_train
            ), (
                f'{self.steps_per_epoch} != '
                f'{self.n_train_samples_per_epoch} / {self.batch_size_train}'
            )
        return self


def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    train_set = ICRDataset('train', config)

    cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = 0.

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(*(train_set[:]))):
        

        train_indices: typing.List[int]
        train_indices: typing.List[int]
        print(f'\n --------------- Fold: {fold} ------------------')
        train_loader = train_set.make_subset(train_indices, 'train').dataloader
        valid_loader = train_set.make_subset(valid_indices, 'test').dataloader
        config = Config(
            **{
                **config.model_dump(),
                'log_dir': os.path.join(cv_log_dir, f'fold-{fold}'),
                'steps_per_epoch': len(train_loader),
            }
        )
        config.display()

        model = ICRModel(config)
        model_utils: ICRModelUtils = ICRModelUtils.start_new_training(model, config)

        history = model_utils.train(config.epochs, train_loader, valid_loader)

        best_valid_loss = history.get_best_criterion()
        cv_loss += best_valid_loss.value

        model_utils.plot_history()
        return
    
    cv_loss /= k

    print(f'cv_loss = {cv_loss:.6f}')
    with open(os.path.join(cv_log_dir, 'cv_loss.txt'), 'w', encoding='utf8') as fout:
        fout.write(f'{cv_loss}')
    return


def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = Config(
        epochs=100,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        
    )
    cross_validation(5, config)

    # dir_path = 'log/cv-20230727T00-07-03/fold-0/20230727T00-07-03'
    # model = ICRModel(config)
    # utils = ICRModelUtils.load_last_checkpoint(model, dir_path)
    # utils.plot_history()
    # config = utils.config
    # config.display()
    return

if __name__ == '__main__':
    main()
