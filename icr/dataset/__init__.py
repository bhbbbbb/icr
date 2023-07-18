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
        split: Literal['train', 'test', 'infer'],
        *_,
        **kwargs,
    ):
        self.split = split
        return

    @property
    def dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """
        pass

    def load(self):
        pass

    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_
        """
        pass
    