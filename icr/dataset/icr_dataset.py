from __future__ import annotations
# import os
from typing import Literal, get_args, Tuple, ClassVar, Dict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_utils.config import BaseConfig

ModeT = Literal['train', 'valid', 'infer']
class ICRDatasetConfig(BaseConfig):

    batch_size_train: int
    batch_size_eval: int

    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'
    persistent_workers: bool = False
    pin_memory: bool =True
    num_workers: int

    class_sample_weights: Tuple[float, float] = (1, 1)
    """weights for sampling each case

    E.g. (1., 1.) indicates the sampler would sample half class0 and half class1
    """

    n_train_samples_per_epoch: int = 1024
    """# of train samples per epoch"""


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
        self.df = ICRDataset._load_df(path)
        return

    @classmethod
    def _load_df(cls, path: str):

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
        #         stratify=df['Class'],
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
        df['EJ'] = df['EJ'].map({'A': 0, 'B': 1})
        scaler = StandardScaler()
        df.iloc[:, df.columns != 'Class'] = scaler.fit_transform(
            df.iloc[:, df.columns != 'Class'])
        
        cls._df_cache[path] = df
        return df

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

        labels = self.df['Class']
        class_weights = self.config.class_sample_weights / labels.value_counts()
        sample_weights: pd.Series = labels.map(class_weights)
        sampler = WeightedRandomSampler(
            sample_weights.to_numpy(),
            self.config.n_train_samples_per_epoch,
            replacement=True
        )

        return DataLoader(self,
                            batch_size=batch_size,
                            num_workers=self.config.num_workers,
                            persistent_workers=self.config.persistent_workers,
                            pin_memory=self.config.pin_memory,
                            drop_last=True,
                            sampler=sampler)


    def make_subset(self, indices: list, mode: ModeT) -> ICRDataset:
        """Get Subset. Define a way to split the dataset with the given indices.

        This method is used by k-fold cross validation.
        
        Args:
            indicies (slice): _description_

        Returns:
            ICRDataset: _description_
        """
        assert self.mode == 'train'
        subset = ICRDataset('train', self.config)
        subset.mode = mode
        subset.df = subset.df.iloc[indices]
        return subset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        """

        Args:
            index: int: 
        
        Returns:
            Tuple[features, labels]
            if index is int:
                features(float): (n_features, )
                labels(int): 
        """
        row = self.df.iloc[index]
        if self.mode == 'infer':
            return row.to_numpy()

        features = row.drop('Class')
        label = row['Class']
        return features.to_numpy(), int(label)
