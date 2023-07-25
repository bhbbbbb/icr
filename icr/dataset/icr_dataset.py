from __future__ import annotations
from typing import Literal
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def __init__(
        self,
        mode: Literal['train', 'test', 'infer'],
        batch_size=64,
        persistent_workers=False,
        pin_memory=True,
        num_workers=8,
        *_,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.pin_memory = pin_memory,
        self.num_workers = num_workers
        assert mode in ['train', 'test', 'infer']
        # trainset -> 85% of train.csv -> ratio of train & valid according to k-fold
        # testset -> 15% of train.csv
        self.df = pd.read_csv(
            '/Users/kouyasushi/Desktop/icr/icr/dataset/train.csv')
        self.mode = mode
        self.persistent_workers = persistent_workers
        self.df.drop(columns=['Id'], inplace=True)
        self.missing_value(self.df)
        self.df['EJ'] = self.df['EJ'].map({'A': 0, 'B': 1})
        self.data = self.df.iloc[:, :56]
        self.label = self.df['Class']

        # using the train test split function
        X_train, X_test, y_train, y_test = train_test_split(self.data,
                                                            self.label,
                                                            random_state=104,
                                                            test_size=0.15,
                                                            shuffle=True)

        scaler = StandardScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])
        n1 = sum(y_test)  # 類別1的樣本數
        n0 = len(y_test) - n1  # 類別0的樣本數
        N = n0 + n1  # 總樣本數
        self.weights = []
        weight_0 = N / (2 * n0)
        weight_1 = N / (2 * n1)
        for label in y_train:
            if label == 1:
                self.weights.append(weight_1)
            else:
                self.weights.append(weight_0)

        return

    @property
    def dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """
        sampler = WeightedRandomSampler(self.weights,
                                        self.batch_size,
                                        replacement=False)
        if self.mode == 'train':
            return DataLoader(self,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              persistent_workers=self.persistent_workers,
                              pin_memory=self.pin_memory,
                              drop_last=(self.mode != 'train'),
                              sampler=sampler)
        return DataLoader(self,
                          batch_size=self.batch_size,
                          shuffle=(self.mode == 'train'),
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          drop_last=(self.mode != 'train'))

    def make_subset(self, indices: list, mode: Literal['train', 'test',
                                                       'infer']) -> ICRDataset:
        """Get Subset. Define a way to split the dataset with the given indices.

        This method is used by k-fold cross validation.
        
        Args:
            indicies (slice): _description_

        Returns:
            ICRDataset: _description_
        """
        subset = ICRDataset(mode)
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
        features = self.data.iloc[index]
        labels = self.label.iloc[index]
        return features.to_numpy(), labels

    def missing_value(self, df):
        df.fillna(df.mean(), inplace=True)
        return
