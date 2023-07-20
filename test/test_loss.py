import torch
from torch import Tensor
from icr.model_utils.loss import BlancedLogLoss

loss_fn = BlancedLogLoss([1, 1])

y_true = Tensor([1, 1, 0, 0, 0]).to(torch.long)
loss = loss_fn.forward(y_true, Tensor([1] * len(y_true)))
print(loss) #17.3
loss = loss_fn.forward(y_true, Tensor([0] * len(y_true)))
print(loss) #17.3
loss = loss_fn.forward(y_true, Tensor([0.5] * len(y_true)))
print(loss) #0.693
