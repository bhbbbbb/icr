from __future__ import annotations
from typing import Literal
from torch.utils.data import Dataset, DataLoader

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
        *_,
        **kwargs,
    ):
        assert mode in ['train', 'test', 'infer']
        self.mode = mode
        return

    @property
    def dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """
        pass

    def make_subset(self) -> ICRDataset:
        """Get Subset. Define a way to split the dataset with the given indices.

        This method is used by k-fold cross validation.
        
        Args:
            indicies (slice): _description_

        Returns:
            ICRDataset: _description_
        """
        pass


    def __len__(self):...


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
        features = ...
        labels = ...

        return features, labels
    