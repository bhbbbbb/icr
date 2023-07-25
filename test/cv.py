#pylint: disable=all
import os
import typing

from sklearn.model_selection import KFold

from model_utils import formatted_now

from icr.dataset.foo_dataset import ICRDataset as ICRDataset
from icr.models import ICRModel
from icr.model_utils import ICRModelUtils
from icr.model_utils.config import ICRModelUtilsConfig


class Config(ICRModelUtilsConfig):
    epochs: int = 3
    loss_class_weights: typing.Tuple[float, float] = [1., 1.]
    save_best: bool = True
    epochs_per_checkpoint: int = 0

def train(config):

    train_set = ICRDataset('train')

    return
    # model = ICRModel()
    # model_utils = ICRModelUtils.start_new_training(model, config)
    return

def cross_validation(k: int):
    train_set = ICRDataset('train')

    cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = 0.

    spliter = KFold(n_splits=k, shuffle=True)
    
    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_set.df)):
        
        print(f'\n --------------- Fold: {fold} ------------------')
        # input('continue...')
        # train_set = train_set.make_subset(train_indices, 'train')
        # train_loader = train_set.dataloader
        train_loader = train_set.make_subset(train_indices, 'train').dataloader
        valid_loader = train_set.make_subset(valid_indices, 'test').dataloader
        config = Config(
            steps_per_epoch = len(train_loader),
            log_dir = os.path.join(cv_log_dir, f'fold-{fold}'),
        )
        config.display()

        model = ICRModel(config)
        model_utils: ICRModelUtils = ICRModelUtils.start_new_training(model, config)

        history = model_utils.train(config.epochs, train_loader, valid_loader)

        best_valid_loss = history.get_best_criterion()
        cv_loss += best_valid_loss.value
    
    cv_loss /= k

    print(f'cv_loss = {cv_loss:.6f}')
    with open(os.path.join(cv_log_dir, 'cv_loss.txt'), 'w', encoding='utf8') as fout:
        fout.write(f'{cv_loss}')
    return


def main():

    # cross_validation(5)

    model = ICRModel('')
    dir_path = 'log/cv-20230723T17-16-27/fold-4/20230723T17-16-30'
    utils = ICRModelUtils.load_last_checkpoint(model, dir_path)
    config = utils.config
    config.display()
    return

if __name__ == '__main__':
    main()
