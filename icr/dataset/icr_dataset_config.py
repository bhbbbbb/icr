from typing import Tuple, Optional, Literal, Union

from model_utils.config import BaseConfig
from pydantic import model_validator

class ICRDatasetConfig(BaseConfig):

    batch_size_train: int
    batch_size_eval: int

    train_csv_path: str = 'train.csv'
    test_csv_path: str = 'test.csv'
    greeks_csv_path: str = 'greeks.csv'
    persistent_workers: bool = False
    pin_memory: bool = True
    num_workers: int = 0

    standard_scale_enable: bool = True

    class UnderSamplingConfig(BaseConfig):
        class_sample_weights: Tuple[float, float] = (1, 1)
        """weights for sampling each case

        E.g. (1., 1.) indicates the sampler would sample half class0 and half class1
        """

        n_train_samples_per_epoch: int = 1024
        """# of train samples per epoch"""
    
    under_sampling_config: Optional[UnderSamplingConfig] = None
    """pass None to NOT using under-sampling"""

    class OverSamplingConfig(BaseConfig):

        sampling_strategy: Union[float, dict, Literal['auto']] = 1.

        method: Literal['smote', 'random']

    # over_sampling_strategy: Optional[Union[float, dict]] = None

    over_sampling_config: Optional[OverSamplingConfig] = None
    """pass None to NOT using over-sampling"""


    @model_validator(mode='after')
    def check_sampling_mutual_exclusive(self):
        assert bool(self.under_sampling_config) != bool(self.over_sampling_config), (
                'got both under-samping_config and over_sampling config'
            )
        return self
    
    labels: Literal['class', 'alpha'] = 'class'

    epsilon_as_feature: bool = False

