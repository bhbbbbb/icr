from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BlancedLogLoss(nn.Module):
    """BlancedLogLoss"""

    class_weights: Tensor
    def __init__(self, class_weights: Tuple[float, float], device = None):
        super().__init__()

        self.device = device
        self.class_weights = Tensor(class_weights).to(device)
        return

    def forward(self, y_true: Tensor, y_pred: Tensor):
        """

        Args:
            y_true (Tensor): (batch, )
            y_pred (Tensor): (batch, )

        Returns:
            _type_: _description_
        """
        assert not y_true.is_floating_point(), f'y_true.dtype = {y_true.dtype}'
        nc = Tensor([(y_true == 0).sum(), (y_true == 1).sum()]).to(self.device)

        assert nc.detach().sum() == len(y_true)

        weights = (1 / nc).mul(self.class_weights)
        sample_weights = weights[y_true]

        y_pred = y_pred.to(torch.float64).clip(1e-15, 1 - 1e-15)
        losses = F.binary_cross_entropy(y_pred, y_true.to(torch.float64), reduction='none')
        losses = losses.mul(sample_weights)
        return losses.sum()
