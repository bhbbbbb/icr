#pylint: disable=all
import os
import warnings
import typing

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


from icr.dataset import ICRDataset, ICRDatasetConfig
# from icr.classifier import ICRXGBClassfier
from icr.classifier import ICRClassifier
from termcolor import cprint

from icr.tools import seed_everything
from icr.tools.metrics import balanced_log_loss, compare_with_color, precision_recall
from icr.post_analysis import pred_analysis #, post_analysis

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
    profiles: typing.Sequence[str]
    tabpfn_config: dict
    prediction_analysis: bool = False



def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    # cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = np.zeros(k)
    ps = np.zeros(k)
    rs = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    train_class = train_set.class_ser
    cat_col_index = train_set.cat_column_locate
    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_class, train_class)):
        
        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')

        train_subset = train_subset.new_over_sampled_dataset()
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)

        predictions = np.zeros(len(x_valid), dtype=np.float64)
        for profile in config.profiles:
            y_train = alpha_train if 'm' in profile else class_train
            y_valid = alpha_valid if 'm' in profile else class_valid

            classifier = ICRClassifier(
                profile,
                seed,
                cat_col_index=cat_col_index,
                class_labels=class_train,
                tab_config=config.tabpfn_config,
            )
            classifier.fit(x_train, y_train, x_valid, y_valid)

            prediction = classifier.predict_proba(x_valid)
            predictions += prediction
        
        predictions /= len(config.profiles)
        eval_loss = balanced_log_loss(predictions, class_valid)
        if config.prediction_analysis:
            pred_analysis(predictions, alpha_valid, f'{hex(seed)}-{fold}')
        print(f'eval_loss = {eval_loss}')
        precision, recall, wrongs = precision_recall(predictions, class_valid)
        precision_str, recall_str = compare_with_color(precision, recall)
        print(f'precision: {precision_str}, recall: {recall_str}, wrongs: {wrongs}')
        cv_loss[fold] = eval_loss
        ps[fold] = precision
        rs[fold] = recall
        
    
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    return cv_loss, ps, rs


def main():

    config = Config(
        # epochs=100,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        tabpfn_config={'device': 'cuda:0', 'base_path': dataset_dir},
        under_sampling_config=None,
        over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy=.5,
            # sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47}, # k = 5
            # sampling_strategy={0: 459, 1: 110, 2: 33, 3: 53}, # k = 10
            sampling_strategy={0: 1., 1: 3., 2: 2., 3: 2.}, # k = 10
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
        # xgb_profiles=[f'xgb{i}' for i in range(7)],
        # xgb_profile=['xgb1', 'xgb2', 'xgb3'],
        prediction_analysis=True,
        profiles=['lgb1'],
    )
    config.display()
    k = 10
    s = 5
    seeds = [0xAAAAAA + i for i in range(s)]
    cv_loss = np.zeros((s, k))
    ps = np.zeros((s, k))
    rs = np.zeros((s, k))

    for idx, seed in enumerate(seeds):
        cv_loss[idx, :], ps[idx, :], rs[idx, :] = cross_validation(k, config, seed)


    print('-' * 60)
    print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean(axis=1)}, {cv_loss.std(axis=1)}')
    cprint(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}', on_color='on_cyan')
    # print('precisions')
    # print(ps.tolist())
    # print('recalls')
    # print(rs.tolist())



if __name__ == '__main__':
    main()
