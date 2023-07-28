from typing import Tuple, Optional
from model_utils import ModelUtilsConfig

from ..tools import is_on_kaggle

class ICRModelUtilsConfig(ModelUtilsConfig):
    """configs for ICRModelUtils"""

    device: str = 'cuda:0'
    """Device to use, cpu or gpu"""

    weight_decay: float = 0.0001

    epochs_per_checkpoint: int = 0
    """num of epochs per checkpoints

        Examples:
            1: stand for save model every epoch
            0: for not save until finish
    """

    # log_dir: ModelUtilsConfig.MutableField[str] = 'log'
    log_dir: str = 'log'
    """dir for saving checkpoints and log files"""

    logging: bool = not is_on_kaggle()
    """whether log to file 'log.log'. It's useful to turn this off when inference on kaggle"""

    epochs_per_eval: int = 1
    """Number of epochs per evalution"""

    early_stopping_rounds: int = 10

    save_n_best: int = 0
    """set True to save every time when the model reach best valid score."""

    progress_bar: bool = True

    loss_class_weights: Tuple[float, float] = (1, 1)





    # -------------- LR configs --------------------
    learning_rate: float = 1e-4
    pct_start: float = 0.1
    div_factor: float = 1e5
    epochs: int
    steps_per_epoch: Optional[int] = None
