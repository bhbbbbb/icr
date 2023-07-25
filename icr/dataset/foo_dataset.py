from __future__ import annotations
from typing import Literal


import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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
    ):
        """_summary_

        Args:
            mode (Literal[&#39;train&#39;, &#39;test&#39;, &#39;infer&#39;]): 
            'train: train.csv
            'test: train.csv #valid
            'infer: test.csv
        """
        self.mode = mode
        self.df = pd.DataFrame(np.zeros((617, 57)))
        self.x = self.df[range(56)]
        self.y = self.df[56].astype(int)
        return


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """

        Args:
            index (int, list): 
        
        Returns:
            Tuple[features, labels]
            if index is list:
                features(float): (batch_size, n_features)
                labels(int): (batch_size, )
            if index is int:
                features(float): (n_features, )
                labels(int): 
        """
        features = self.x.iloc[index]
        labels = self.y.iloc[index]
        return features.to_numpy(), labels


    def make_subset(self, indices: list, mode: Literal['train', 'test', 'infer']) -> ICRDataset:
        """Get Subset. Define a way to split the dataset with the given indices.

        This method is used by k-fold cross validation.
        
        Args:
            indicies (slice): _description_

        Returns:
            ICRDataset: _description_
        """
        subset = ICRDataset(mode)
        subset.df = pd.DataFrame(np.zeros((len(indices), 57)))
        subset.x = subset.df[range(56)]
        subset.y = subset.df[56].astype(int)
        return subset

    @property
    def dataloader(self) -> DataLoader:
        """define ways to produce dataloader.         """

        # class1_weight: float = 2. # config

        # weights = self.y.map(lambda label: class1_weight if label == 1.0 else 1.)
        
        # num_samples = (self.y == 1).sum() + (self.y == 0).sum() * 1 / class1_weight

        # # num_samples: set to make each (label == 1) be sampled once in expectation
        # sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

        return DataLoader(
            self,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            # sampler=sampler,
        )
