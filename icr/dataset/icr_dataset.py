from __future__ import annotations
# import os
# import math
from datetime import datetime
import warnings
from typing import (
    Literal, get_args, Tuple, ClassVar, Sequence, Union, overload
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC, RandomOverSampler

from .icr_dataset_config import ICRDatasetConfig
from .df_cache import DfCache

ModeT = Literal['train', 'valid', 'infer']
CAT_COL = 'EJ'
CLASS_COL = 'Class'
EPSILON_COL = 'Epsilon'


class ICRDataset(Dataset):
    """ICR Dataset

    Responsibilities:
        1. Load data from csv files
            - split them to train, test sets
            - 85:15
        1. Preprocessing
            - missing value
            - categorical encoding
        1. Dataloader 
            - **undersampling**

    """

    _df_cache: ClassVar[DfCache] = DfCache()

    def __init__(
        self,
        mode: ModeT,
        config: ICRDatasetConfig,
    ):
        assert mode in get_args(ModeT)
        self.config = config
        self.mode = mode
        
        self.train_df, self.class_ser, self.alpha_ser, self.test_df = ICRDataset._load_df(
            config.train_csv_path,
            config.greeks_csv_path,
            config.test_csv_path,
            config.standard_scale_enable,
            config.epsilon_as_feature,
        )
        

        # self.over_sampler = self._load_over_sampler(
        #     config.over_sampling_config,
        #     self.df.columns.get_loc(CAT_COL),
        # )
        return

    @property
    def cat_column_locate(self):
        return self.train_df.columns.get_loc(CAT_COL)



    @classmethod
    def _load_df(
        cls,
        train_path: str,
        meta_path: str,
        test_path: str,
        standard_scale_enable: bool,
        epsilon_as_feature: bool,
    ):

        train_df = cls._df_cache.get_copy(train_path)
        meta_df = cls._df_cache.get_copy(meta_path)
        test_df = cls._df_cache.get_copy(test_path)

        train_df.drop(columns=['Id'], inplace=True)
        train_df[CAT_COL] = train_df[CAT_COL].map({'A': 0, 'B': 1})
        test_df.drop(columns=['Id'], inplace=True)
        test_df[CAT_COL] = test_df[CAT_COL].map({'A': 0, 'B': 1})
        class_df = train_df[CLASS_COL]
        train_df = train_df.drop([CLASS_COL], axis=1)

        alpha_values = ['A', 'B', 'D', 'G']
        alpha_df = meta_df.Alpha.map({v: i for i, v in enumerate(alpha_values)})

        if epsilon_as_feature:
            epsilon = meta_df[EPSILON_COL]
            def mapper(date_str: str):
                if date_str == 'Unknown':
                    return np.nan
                return datetime.strptime(date_str, '%m/%d/%Y').toordinal()

            epsilon = epsilon.map(mapper)
            train_df.insert(len(train_df.columns), EPSILON_COL, epsilon)

            # --------- test csv ----------------
            test_epsilon = [epsilon.max() for _ in range(len(test_df))]
            test_df.insert(len(test_df.columns), EPSILON_COL, test_epsilon)

        
        imputer = SimpleImputer(strategy='median')

        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                # category=DeprecationWarning,
                message=(
                    '.*will attempt to set the values inplace instead of always '
                    'setting a new array. '
                    'To retain the old behavior, use either.*'
                ),
            )
            train_df.iloc[:, :] = imputer.fit_transform(train_df)
            test_df.iloc[:, :] = imputer.transform(test_df)

            if standard_scale_enable:
                scaler = StandardScaler()
                train_cat_col = train_df[CAT_COL].copy()
                test_cat_col = test_df[CAT_COL].copy()
                train_df.iloc[:, :] = scaler.fit_transform(train_df)
                test_df.iloc[:, :] = scaler.transform(test_df)
                train_df[CAT_COL] = train_cat_col
                test_df[CAT_COL] = test_cat_col
        
        return train_df, class_df, alpha_df, test_df

    @classmethod
    def _get_under_sampler(
        cls,
        labels: pd.Series,
        config: ICRDatasetConfig.UnderSamplingConfig
    ):
        class_weights = config.class_sample_weights / labels.value_counts()
        sample_weights: pd.Series = labels.map(class_weights)
        return WeightedRandomSampler(
            sample_weights.to_numpy(),
            config.n_train_samples_per_epoch,
            replacement=True
        )

    @property
    def dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """

        batch_size = (
            self.config.batch_size_train if self.mode == 'train'
            else
            self.config.batch_size_eval
        )

        if self.mode != 'train':
            return DataLoader(self,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=self.config.num_workers,
                            persistent_workers=self.config.persistent_workers,
                            pin_memory=self.config.pin_memory,
                            drop_last=False)

        # mode == 'train'
        if self.config.under_sampling_config is not None:
            assert self.config.labels == 'class'
            labels = self.class_ser
            sampler = self._get_under_sampler(labels, self.config.under_sampling_config)

            return DataLoader(self,
                                batch_size=batch_size,
                                num_workers=self.config.num_workers,
                                persistent_workers=self.config.persistent_workers,
                                pin_memory=self.config.pin_memory,
                                drop_last=True,
                                sampler=sampler)


        return DataLoader(self.new_over_sampled_dataset(),
                            batch_size=batch_size,
                            num_workers=self.config.num_workers,
                            persistent_workers=self.config.persistent_workers,
                            pin_memory=self.config.pin_memory,
                            drop_last=True,
                            shuffle=True)

    def new_over_sampled_dataset(
        self,
    ) -> OverSampledDataset:
        assert self.mode == 'train'

        os_config = self.config.over_sampling_config
        assert os_config.method in ['smote', 'random']

        sampling_strategy = os_config.sampling_strategy

        if isinstance(sampling_strategy, dict):
            assert self.config.labels == 'alpha'
            sampling_strategy = os_config.sampling_strategy.copy()

            if isinstance(next(iter(sampling_strategy.values())), float):
                bins = self.alpha_ser.value_counts()
                for k, v in sampling_strategy.items():
                    sampling_strategy[k] = int(bins[k] * v)

        sampler = (
            SMOTENC(
                [self.cat_column_locate],
                sampling_strategy=sampling_strategy,
            )
            if os_config.method == 'smote' else
            RandomOverSampler(sampling_strategy=sampling_strategy)
        )

        x, y = sampler.fit_resample(*(self[:]))

        return OverSampledDataset(x, y)
    
    def make_subset(self, indices: list, mode: ModeT) -> ICRDataset:
        """Get Subset. Define a way to split the dataset with the given indices.

        This method is used by k-fold cross validation.
        
        Args:
            indicies (slice): _description_

        Returns:
            ICRDataset: _description_
        """
        assert self.mode == 'train'
        assert mode in get_args(ModeT)
        subset = ICRDataset('train', self.config)
        subset.mode = mode
        subset.train_df = self.train_df.iloc[indices]
        subset.class_ser = self.class_ser.iloc[indices]
        subset.alpha_ser = self.alpha_ser.iloc[indices]
        return subset

    def __len__(self):
        return len(self.train_df) if self.mode != 'infer' else len(self.test_df)

    
    @overload
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """When mode != 'infer'"""

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        """When mode == 'infer'"""

    @overload
    def __getitem__(self, indices: Union[Sequence[int], slice]) -> Tuple[np.ndarray, np.ndarray]:
        """When mode != 'infer'"""

    @overload
    def __getitem__(self, indices: Union[Sequence[int], slice]) -> np.ndarray:
        """When mode == 'infer'"""


    def __getitem__(self, index: Union[int, Sequence[int]]):
        """

        Args:
            index: int: 
        
        Returns:
            Tuple[features, labels]
            if index is int:
                features(float): (n_features, )
                labels(int): 
        """
        index = [index] if isinstance(index, int) else index
        if self.mode == 'infer':
            return self.test_df.iloc[index].to_numpy()

        features = self.train_df.iloc[index]
        labels = (
            self.class_ser[index]
            if self.config.labels == 'class' else
            self.alpha_ser[index]
        )


        return features.to_numpy().squeeze(), labels.to_numpy().squeeze()

class OverSampledDataset(Dataset):
    """Simple wrapper dataset for SMOTE (since smote regenerate date every time)"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        return
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


        # def _train_test_split(df: pd.DataFrame, cache_dir: str):
        #     """load split indices if split.csv in cache_dir. Else
        #     split train.csv and save the result into

        #     Args:
        #         df (pd.DataFrame): _description_
        #         cache_dir (str): _description_

        #     Returns:
        #         _type_: _description_
        #     """
        #     train_path = os.path.join(cache_dir, 'train_split.csv')
        #     valid_path = os.path.join(cache_dir, 'valid_split.csv')

        #     if os.path.isfile(train_path) and os.path.isfile(valid_path):
        #         train_idices = pd.read_csv(train_path)['indices']
        #         valid_idices = pd.read_csv(valid_path)['indices']
        #         return train_idices, valid_idices

        #     train_df, valid_df = train_test_split(
        #         df,
        #         random_state=104,
        #         test_size=0.15,
        #         shuffle=True,
        #         stratify=df[CLASS_COL],
        #     )
        #     train_df: pd.DataFrame
        #     valid_df: pd.DataFrame
        #     train_df.index.to_frame(name='indices').to_csv(train_path, index=False)
        #     valid_df.index.to_frame(name='indices').to_csv(valid_path, index=False)
        #     return train_df.index, valid_df.index