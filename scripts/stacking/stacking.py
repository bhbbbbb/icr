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
from icr.classifier import ICRClassifier, ICREnsembleClassifier

from icr.tools import seed_everything, is_on_kaggle, alpha_to_class
from icr.tools.metrics import balanced_log_loss, precision_recall, compare_with_color
from icr.tools.ensemble import ensemble_proba
from icr.classifier.params import profiles as all_profile

from icr.post_analysis import pred_analysis #, post_analysis
from .base import backbone_fit_predict_proba_table, StackConfig



class Config(ICRDatasetConfig, StackConfig):
    batch_size_train: int = 64
    batch_size_eval: int = 128
    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'

    persistent_workers: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4

    # labels: typing.Literal['class', 'alpha'] = 'alpha'
    # standard_scale_enable: bool = False
    stacking_profiles: typing.Sequence[str]
    inner_profiles: typing.Sequence[str]
    inner_k: int
    tabpfn_config: dict
    passthrough: bool
    prediction_analysis: bool = False
    outer_k: int
    n_seeds: int


def post_process(predictions: np.ndarray):
    # over_positive = predictions > .95
    # over_negative = predictions < .05
    # predictions[(predictions > .65) & (predictions < .95)] = .95
    return predictions.clip(min=.001)



def cross_validation(k: int, config: Config, seed: int = 0xAAAA):

    seed_everything(seed)
    train_set = ICRDataset('train', config)

    spliter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_class = train_set.class_ser
    cat_col_index = train_set.cat_column_locate

    for fold, (train_indices, valid_indices) in enumerate(spliter.split(train_class, train_class)):

        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')
        infer_set = ICRDataset('infer', config)
        train_class = train_subset.class_ser

        
        oof_pred_table, valid_pred_table, infer_pred_table =\
            backbone_fit_predict_proba_table(config, train_subset, valid_subset, infer_set, seed)

        train_predicted_df = pd.DataFrame(oof_pred_table)
        valid_predicted_df = pd.DataFrame(valid_pred_table)
        infer_predicted_df = pd.DataFrame(infer_pred_table)


        if config.passthrough:
            train_subset.train_df = pd.concat(
                [train_subset.train_df.reset_index(drop=True), train_predicted_df], axis=1)
            train_subset = train_subset.new_over_sampled_dataset()

            valid_subset.train_df = pd.concat(
                [valid_subset.train_df.reset_index(drop=True), valid_predicted_df], axis=1)
            infer_set.test_df = pd.concat(
                [infer_set.test_df.reset_index(drop=True), infer_predicted_df], axis=1)
        else:
            train_subset.train_df = train_predicted_df
            valid_subset.train_df = valid_predicted_df
            infer_set.train_df = infer_predicted_df
        
        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        
        if config.stacking_profiles:
            # do stacking
            classifier = ICREnsembleClassifier(
                config.stacking_profiles,
                seed,
                cat_col_index=cat_col_index,
                class_labels=alpha_train,
                tab_config=config.tabpfn_config,
            )
            classifier.fit(
                x_train,
                alpha_train=alpha_train,
                x_valid=x_valid,
                alpha_valid=alpha_valid,
            )

            valid_pred = classifier.predict_proba(x_valid)
            infer_pred = classifier.predict_proba(infer_set[:])
            
        else:
            # average the outputs of inner models
            def predict_inner_probas(inner_profiles, pred_table):
                probe = 0
                for profile in inner_profiles:
                    if 'm' in profile:
                        yield (1 - pred_table[:, probe])
                        probe += 4
                    else:
                        yield (pred_table[:, probe])
                        probe += 1

            valid_pred = ensemble_proba(
                predict_inner_probas(config.inner_profiles, valid_pred_table)
            )
            infer_pred = ensemble_proba(
                predict_inner_probas(config.inner_profiles, infer_pred_table)
            )

        # infer_pred_post = post_process(infer_pred.copy()) ###
        valid_pred_post = post_process(valid_pred.copy())
        eval_loss = balanced_log_loss(valid_pred, alpha_to_class(alpha_valid))
        eval_loss_post = balanced_log_loss(valid_pred_post, alpha_to_class(alpha_valid))
        before_loss, after_loss = compare_with_color(eval_loss, eval_loss_post, reverse=True)
        print(f'eval_loss (before, after) = ({before_loss}, {after_loss})')
        precision, recall, wrongs = precision_recall(valid_pred, alpha_to_class(alpha_valid))
        precision_str, recall_str = compare_with_color(precision, recall)
        print(f'precision: {precision_str}, recall: {recall_str}, wrongs: {wrongs}')
        if config.prediction_analysis and not is_on_kaggle():
            # post_analysis(predictions, raw_predictions, class_valid, f'{seed}-{fold}')
            pred_analysis(valid_pred, alpha_to_class(alpha_valid), f'{hex(seed)}-{fold}')

        yield eval_loss, infer_pred


def train_inference(config: Config):

    SEED_BASE = 0xAAAAAA
    cv_losses = np.zeros((config.n_seeds, config.outer_k), dtype=np.float64)
    seeds = [SEED_BASE + i for i in range(config.n_seeds)]

    def run_a_seed(seed: int):

        # cv_loss = np.zeros(config.outer_k)

        cv = cross_validation(config.outer_k, config, seed)

        def run():
            for fold, (eval_loss, infer_pred) in enumerate(cv):
                # cv_loss[fold] = eval_loss
                sidx = seed - SEED_BASE
                cv_losses[sidx, fold] = eval_loss
                acc_loss = cv_losses.flatten()[:sidx * config.n_seeds + fold + 1]
                print(
                    f'eval_loss[fold:{fold}-seed:{hex(seed)}] = {eval_loss:.4f}\n'
                    f'accmulated_cv_loss = {acc_loss.mean():.4f}, {acc_loss.std():.4f}'
                )
                yield infer_pred
        
        infer_prediction = ensemble_proba(run())
        return infer_prediction

    infer_prediction = ensemble_proba(run_a_seed(seed) for seed in seeds)

    print('-' * 60)
    print(f'cv_losses = {cv_losses.tolist()}')
    print(f'cv_losses = {cv_losses.mean(axis=1)}, {cv_losses.std(axis=1)}')
    cprint(f'cv_losses = {cv_losses.mean()}, {cv_losses.std()}', on_color='on_cyan')

    return infer_prediction
