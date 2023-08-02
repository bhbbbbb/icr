from typing import Sequence
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model_utils.config import BaseConfig
from icr.classifier import ICRClassifier
from icr.dataset import ICRDataset

class StackConfig(BaseConfig):
    inner_k: int
    inner_profiles: list
    tabpfn_config: dict

def backbone_fit_predict_proba_table(
        config: StackConfig,
        train_set: ICRDataset,
        valid_set: ICRDataset,
        infer_set: ICRDataset,
        seed,
    ):

    assert train_set.config.labels == 'alpha'

    inner_dim = sum(4 if 'm' in profile else 1 for profile in config.inner_profiles)
    oof_prediction = np.zeros((len(train_set), inner_dim), dtype=np.float64)
    valid_prediction = np.zeros((len(valid_set), inner_dim), dtype=np.float64)
    infer_prediction = np.zeros((len(infer_set), inner_dim), dtype=np.float64)

    train_class = train_set.class_ser
    cat_col_index = train_set.cat_column_locate
    spliter = StratifiedKFold(config.inner_k, shuffle=True, random_state=seed)

    for _fold, (train_indices, valid_indices) in\
        enumerate(spliter.split(train_class, train_class)):

        train_subset = train_set.make_subset(train_indices, 'train')
        valid_subset = train_set.make_subset(valid_indices, 'valid')
        train_subset = train_subset.new_over_sampled_dataset()

        x_train, alpha_train = train_subset[:]
        x_valid, alpha_valid = valid_subset[:]
        x_test, _alpha_test = valid_set[:]
        x_infer = infer_set[:]

        valid_pred, (test_pred, infer_pred) = _backbones_fit_predict_proba(
            config.inner_profiles,
            seed,
            cat_col_index=cat_col_index,
            tabpfn_config=config.tabpfn_config,
            x_train=x_train,
            x_valid=x_valid,
            alpha_train=alpha_train,
            alpha_valid=alpha_valid,
            x_test_set=(x_test, x_infer),
        )

        oof_prediction[valid_indices] = valid_pred
        valid_prediction += test_pred
        infer_prediction += infer_pred
    
    valid_prediction /= config.inner_k
    infer_prediction /= config.inner_k

    return oof_prediction, valid_prediction, infer_prediction

def _backbones_fit_predict_proba(
        profiles: Sequence[str],
        seed: int,
        *,
        cat_col_index,
        tabpfn_config,
        x_train: np.ndarray,
        alpha_train: np.ndarray,
        x_valid: np.ndarray,
        alpha_valid: np.ndarray,
        x_test_set: Sequence[np.ndarray],
    ):

    def gen_fitted_classifiers(
            inner_profiles: Sequence[str],
            *,
            alpha_train,
            alpha_valid,
        ):
        class_train = (alpha_train != 0).astype(int)
        class_valid = (alpha_valid != 0).astype(int)
        for profile in inner_profiles:
            y_train = alpha_train if 'm' in profile else class_train
            y_valid = alpha_valid if 'm' in profile else class_valid
            classifier = ICRClassifier(
                profile,
                seed,
                cat_col_index=cat_col_index,
                class_labels=class_train,
                tab_config=tabpfn_config,
            )
            yield classifier.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)

    oof_predictions = []
    test_predictions_set = [[] for _ in range(len(x_test_set))]

    fitted_classifiers = gen_fitted_classifiers(
        profiles,
        alpha_train=alpha_train,
        alpha_valid=alpha_valid,
    )
    for classifier in fitted_classifiers:
        # bgb: shape: (n_samples, )
        # mgb: shape: (n_samples, 4)
        pred = classifier.predict_proba(x_valid, no_reshape=('m' in classifier.profile))
        oof_predictions.append(pred if 'm' in classifier.profile else pred.reshape(-1, 1))
        
        for x_test, test_predictions in zip(x_test_set, test_predictions_set):
            test_pred = classifier.predict_proba(x_test, no_reshape=('m' in classifier.profile))
            test_predictions.append(
                test_pred if 'm' in classifier.profile else test_pred.reshape(-1, 1))

    oof_predictions = np.concatenate(oof_predictions, axis=1)
    test_predictions_set = [
        np.concatenate(test_predictions, axis=1) for test_predictions in test_predictions_set
    ]
    return oof_predictions, test_predictions_set
