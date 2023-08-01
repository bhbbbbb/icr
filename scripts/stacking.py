#pylint: disable=all
import os
import warnings
import typing

from termcolor import cprint, colored
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from icr.dataset import ICRDataset, ICRDatasetConfig
from icr.classifier import ICRClassifier
from icr.model_utils.ensemble import Ensemble

from icr.tools import balanced_log_loss, seed_everything, precision_recall, compare_with_color
from icr.classifier.params import profiles as all_profile


from icr.post_analysis import pred_analysis, post_analysis


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
    stacking_profiles: typing.Sequence[str]
    inner_profiles: typing.Sequence[str]
    inner_over_sampling_config: ICRDatasetConfig.OverSamplingConfig
    passthrough: bool
    model_save_dir: str
    tabpfn_config: dict
    prediction_analysis: bool = False


def post_process(predictions: np.ndarray):
    # over_positive = predictions > .95
    # over_negative = predictions < .05
    # predictions[(predictions > .65) & (predictions < .95)] = .95
    return predictions.clip(min=.001)

def inner_cv(
        train_subset: ICRDataset,
        valid_subset: ICRDataset,
        test_subset: ICRDataset,
        config: Config,
        seed,
    ):

    assert config.labels == 'alpha'
    cat_col_index = train_subset.cat_column_locate
    train_subset = train_subset.new_over_sampled_dataset(config.inner_over_sampling_config)
    x_train, alpha_train = train_subset[:]
    x_valid, alpha_valid = valid_subset[:]
    x_test, _alpha_test = test_subset[:]
    class_train = (alpha_train != 0).astype(int)
    class_valid = (alpha_valid != 0).astype(int)

    oof_predictions = []
    test_predictions = []
    models: typing.List[ICRClassifier] = []
    for profile in config.inner_profiles:
        y_train = alpha_train if 'm' in profile else class_train
        y_valid = alpha_valid if 'm' in profile else class_valid
        classifier = ICRClassifier(
            profile,
            seed,
            cat_col_index=cat_col_index,
            class_labels=class_train,
            tab_config=config.tabpfn_config,
        )
        classifier.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)

        prediction = classifier.predict_proba(x_valid, no_reshape=('m' in profile))
        test_prediction = classifier.predict_proba(x_test, no_reshape=('m' in profile))
        # bgb: (n_samples, )
        # mgb: (n_samples, 4)

        models.append(classifier)
        oof_predictions.append(prediction if 'm' in profile else prediction.reshape(-1, 1))
        test_predictions.append(test_prediction if 'm' in profile else test_prediction.reshape(-1, 1))

    oof_predictions = np.concatenate(oof_predictions, axis=1)
    test_predictions = np.concatenate(test_predictions, axis=1)
    return models, oof_predictions, test_predictions


def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    # cv_log_dir = f'log/cv-{formatted_now()}'
    cv_loss = np.zeros(k)
    cv_loss_post = np.zeros(k)
    # best_loss = 100000.
    # best_modelss = None
    ps = np.zeros(k)
    rs = np.zeros(k)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_class = train_set.class_ser
    cat_col_index = train_set.cat_column_locate

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_class, train_class)):

        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')
        train_class = train_subset.class_ser

        inner_dim = sum(4 if 'm' in profile else 1 for profile in config.inner_profiles)
        modelss: typing.List[typing.List[ICRClassifier]] = []
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
        
        train_predicted_df = pd.DataFrame(oof_predictionss)
        valid_predicted_df = pd.DataFrame(valid_predictions)


        if config.passthrough:
            train_subset.df = pd.concat(
                [train_subset.df.reset_index(drop=True), train_predicted_df], axis=1)
            valid_subset.df = pd.concat(
                [valid_subset.df.reset_index(drop=True), valid_predicted_df], axis=1)
            train_subset = train_subset.new_over_sampled_dataset()
        else:
            train_subset.df = train_predicted_df
            valid_subset.df = valid_predicted_df
        
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)

        
        predictions = np.zeros(len(x_valid), dtype=np.float64)

        stacking_models: typing.List[ICRClassifier] = []
        if config.stacking_profiles:
            # do stacking
            for profile in config.stacking_profiles:
                y_train = alpha_train if 'm' in profile else class_train
                y_valid = alpha_valid if 'm' in profile else class_valid

                cat_col = cat_col_index if config.passthrough else None
                classifier = ICRClassifier(
                    profile,
                    seed,
                    cat_col_index=cat_col,
                    class_labels=class_train,
                    tab_config=config.tabpfn_config,
                )
                classifier.fit(x_train, y_train, x_valid, y_valid)
                stacking_models.append(classifier)

                prediction = classifier.predict_proba(x_valid)
                predictions += prediction

            predictions /= len(config.stacking_profiles)
            
        else:
            # average the outputs of inner models
            probe = 0
            for profile in config.inner_profiles:
                if 'm' in profile:
                    predictions += 1 - valid_predictions[:, probe]
                    probe += 4
                else:
                    predictions += valid_predictions[:, probe]
                    probe += 1

            predictions /= len(config.inner_profiles)

        raw_predictions = predictions.copy()
        predictions = post_process(predictions)

        eval_loss = balanced_log_loss(raw_predictions, class_valid)
        eval_loss_post = balanced_log_loss(predictions, class_valid)

        before_loss, after_loss = compare_with_color(eval_loss, eval_loss_post, reverse=True)
        print(f'eval_loss (before, after) = ({before_loss}, {after_loss})')
        precision, recall, wrongs = precision_recall(predictions, class_valid)
        precision_str, recall_str = compare_with_color(precision, recall)
        print(f'precision: {precision_str}, recall: {recall_str}, wrongs: {wrongs}')
        cv_loss[fold] = eval_loss
        cv_loss_post[fold] = eval_loss_post
        ps[fold] = precision
        rs[fold] = recall

        if config.prediction_analysis:
            # post_analysis(predictions, raw_predictions, class_valid, f'{seed}-{fold}')
            pred_analysis(raw_predictions, class_valid, f'{seed}-{fold}')

        # if fold == k - 1:
        model_save_dir =\
            os.path.join(config.model_save_dir, f'seed-{hex(seed)}', f'f-{fold}')
        os.makedirs(model_save_dir, exist_ok=True)
        for stack_model in stacking_models:
            stack_model.save(model_save_dir, f'stack_{stack_model.profile}')
        for f, models in enumerate(modelss):
            save_dir = os.path.join(model_save_dir, f'in-f-{f}')
            os.makedirs(save_dir, exist_ok=True)
            for model in models:
                model.save(save_dir, model.profile)
    
    # print(f'cv_loss = {cv_loss.tolist()}')
    print(f'cv_loss = {cv_loss.mean()}, {cv_loss.std()}')
    # print(f'cv_loss = {cv_loss_post.tolist()}')
    print(f'cv_loss_post = {cv_loss_post.mean()}, {cv_loss.std()}')
    return cv_loss, ps, rs


def main():
    config = Config(
        # epochs=100,
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        model_save_dir=os.path.join(dataset_dir, 'models'),
        tabpfn_config={'device': 'cuda:0', 'base_path': dataset_dir},
        # tabpfn_config={'device': 'cpu', 'base_path': dataset_dir},
        over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy=.5,
            # sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47}, # k=5
            sampling_strategy={0: 459, 1: 110, 2: 33, 3: 53}, # k = 10
            method='smote',
        ),
        inner_over_sampling_config=Config.OverSamplingConfig(
            # sampling_strategy={0: 327, 1: 79, 2: 24, 3: 38}, # k = 5
            sampling_strategy={0: 414, 1: 99, 2: 30, 3: 48}, # k = 10
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
        inner_profiles=[*(f'lgb{i}' for i in range(1, 4)), *(f'xgb{i}' for i in range(1, 4)), 'tab0', 'mtab1'],
        # inner_profiles=['lgb2', 'xgb2', 'tab0', 'mtab1'],
        # inner_profiles=['lgb1'],
        # inner_profiles=[*(f'lgb{i}' for i in range(1, 5)), *(f'xgb{i}' for i in range(1, 6)), 'tab0', 'mtab1'],
        # stacking_profiles=['lgb1'],
        stacking_profiles=['lgb1', 'lgb2', 'lgb3', 'lgb4'],
        # passthrough=False,
        passthrough=True,
        prediction_analysis=True,
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
