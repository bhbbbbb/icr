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

    """


    def __init__(
        self,
        split: Literal['train', 'test', 'infer'],
        batch_size: int,
        num_workers: int,
        ...,
        **kwargs,
    ):
        return

    @property
    def dataloader(self) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """
        pass
        return DataLoader(self, )

    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_
        """
        pass
