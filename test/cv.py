#pylint: disable=all
import os

from sklearn.model_selection import KFold

from model_utils import formatted_now
from model_utils.base_model_utils import get_logger

from icr.dataset.foo_dataset import ICRDataset as ICRDataset
from icr.models import ICRModel
from icr.model_utils import ICRModelUtils
from icr.model_utils.config import ICRModelUtilsConfig

def train(config):

    train_set = ICRDataset('train')

    return
    # model = ICRModel()
    # model_utils = ICRModelUtils.start_new_training(model, config)
    return

def cross_validation(config: ICRModelUtilsConfig, k: int):
    train_set = ICRDataset('train')

    cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = 0.

    spliter = KFold(n_splits=k, shuffle=True)
    
    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_set.df)):
        
        print(f'\n --------------- Fold: {fold} ------------------')
        # input('continue...')
        train_loader = train_set.make_subset(train_indices, 'train').dataloader
        valid_loader = train_set.make_subset(valid_indices, 'test').dataloader
        config.steps_per_epoch = len(train_loader)
        config.log_dir = os.path.join(cv_log_dir, f'fold-{fold}')
        config.check_and_freeze(freeze=False)
        config.display()

        model = ICRModel(config)
        model_utils = ICRModelUtils.start_new_training(model, config)

        history = model_utils.train(config.epochs, train_loader, valid_loader)

        def selector(s):
            return s['valid_criteria']['loss']
        valid_losses = map(selector, history.history)
        best_valid_loss = min(valid_losses)
        cv_loss += best_valid_loss
    
    cv_loss /= k

    print(f'cv_loss = {cv_loss:.6f}')
    with open(os.path.join(cv_log_dir, 'cv_loss.txt'), 'w', encoding='utf8') as fout:
        fout.write(f'{cv_loss}')
    return


def main():
    config = ICRModelUtilsConfig()
    config.epochs = 3
    config.loss_class_weights = [1, 2]
    config.save_best = True
    config.epochs_per_checkpoint = 0

    cross_validation(config, 5)

    ##
    # config.steps_per_epoch = 10##
    # config.check_and_freeze()
    # model = ICRModel(config)
    # model_utils = ICRModelUtils.load_last_checkpoint(model, config)
    return

if __name__ == '__main__':
    main()
