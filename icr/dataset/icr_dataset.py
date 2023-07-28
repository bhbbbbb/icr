from __future__ import annotations
# import os
from typing import (
    Literal, get_args, Tuple, ClassVar, Dict, Sequence, Union, overload
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Dataset
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC

from .icr_dataset_config import ICRDatasetConfig

ModeT = Literal['train', 'valid', 'infer']
CAT_COL = 'EJ'
CLASS_COL = 'Class'

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

    _df_cache: ClassVar[Dict[str, pd.DataFrame]] = {}

    def __init__(
        self,
        mode: ModeT,
        config: ICRDatasetConfig,
    ):
        assert mode in get_args(ModeT)
        self.config = config
        self.mode = mode
        
        path = config.train_csv_path if mode == 'train' else config.test_csv_path
        self.df = ICRDataset._load_df(path, config.standard_scale_enable)
        self.smote = (
            SMOTENC(
                [self.df.columns.get_loc(CAT_COL)],
                sampling_strategy=config.smote_strategy
            )
            if config.smote_strategy is not None else
            None
        )
        return

    @classmethod
    def _load_df(cls, path: str, standard_scale_enable: bool):

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

        df = cls._df_cache.get(path, None)
        if df is not None:
            return df.copy()

        df = pd.read_csv(path)
        df.drop(columns=['Id'], inplace=True)
        df.fillna(df.mean(), inplace=True)
        df[CAT_COL] = df[CAT_COL].map({'A': 0, 'B': 1})
        
        if standard_scale_enable:
            scaler = StandardScaler()
            df.iloc[:, df.columns != CLASS_COL] = scaler.fit_transform(
                df.iloc[:, df.columns != CLASS_COL])
            
            cls._df_cache[path] = df
        return df

    @classmethod
    def _get_under_sampler(
        cls,
        labels: pd.Series,
        config: ICRDatasetConfig.UnderSamplingConfig
    ):
        # labels = self.df[CLASS_COL]
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
            labels = self.df[CLASS_COL]
            sampler = self._get_under_sampler(labels, self.config.under_sampling_config)

            return DataLoader(self,
                                batch_size=batch_size,
                                num_workers=self.config.num_workers,
                                persistent_workers=self.config.persistent_workers,
                                pin_memory=self.config.pin_memory,
                                drop_last=True,
                                sampler=sampler)


        return DataLoader(self.new_smote_dataset(),
                            batch_size=batch_size,
                            num_workers=self.config.num_workers,
                            persistent_workers=self.config.persistent_workers,
                            pin_memory=self.config.pin_memory,
                            drop_last=True,
                            shuffle=True)

    def new_smote_dataset(self) -> SmoteDataset:
        assert self.mode == 'train'
        assert self.config.smote_strategy and self.smote is not None
        x, y = self.smote.fit_resample(*(self[:]))
        return SmoteDataset(x, y)
    
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
        subset.df = subset.df.iloc[indices]
        return subset

    def __len__(self):
        return len(self.df)

    
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
        row = self.df.iloc[index]
        if self.mode == 'infer':
            return row.to_numpy()

        features = row.drop(CLASS_COL, axis=1)
        label = row[CLASS_COL]

        return features.to_numpy().squeeze(), label.to_numpy().squeeze()

class SmoteDataset(Dataset):
    """Simple wrapper dataset for SMOTE (since smote regenerate date every time)"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        return
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
