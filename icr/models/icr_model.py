from torch import nn
from torch.nn import functional as F
from pydantic import field_validator
from model_utils.config import BaseConfig

class ICRModelConfig(BaseConfig):

    num_features: int = 56
    num_targets: int = 1

    # channel_base_dim: int = 128
    # hidden_size: int = 2048
    channel_base_dim: int = 64
    hidden_size: int = 1024
    dropout: float = .3

    @field_validator('channel_base_dim', 'hidden_size')
    @classmethod
    def is_power_of_2(cls, v):
        """check v is power of 2"""
        if v > 0 and v & (v - 1) == 0:
            return v
        raise ValueError(f'value {v} is not power of 2')

class ICRModel(nn.Module):
    """ICR model

    Credit: baosenguo @ github
        - https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """

    def __init__(self, config: ICRModelConfig):
        super().__init__()
        # cha_1 = 256
        # cha_2 = 512
        # cha_3 = 512
        cha_1 = config.channel_base_dim
        cha_2 = cha_1 * 2
        cha_3 = cha_1 * 2

        dim_per_channel = config.hidden_size // cha_1 # dim per channel (> 8)
        cha_po_1 = dim_per_channel // 2
        cha_po_2 = cha_po_1 // 2 * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = dim_per_channel
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(config.num_features)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(config.num_features, config.hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(config.dropout)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(config.dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(config.dropout + .2)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(config.dropout + .1)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(config.dropout + .1)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, config.num_targets))

    def forward(self, x):

        # (B, features)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        # (B, hidden)

        x = x.reshape(x.shape[0],self.cha_1,
                        self.cha_1_reshape)
        # (B, cha_1, -1)


        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        # (B, cha_2, -1)

        x = self.ave_po_c1(x)

        # (B, cha_2, cha_po_1)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))

        # (B, cha_2, cha_po_1)

        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        # (B, cha_2, cha_po_1)

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))

        # (B, cha_3, cha_po_1)

        x =  x * x_s


        x = self.max_po_c2(x)

        # (B, cha_3, cha_po_1 / 2)

        x = self.flt(x)

        # (B, cha_3 * cha_po_1 / 2) == (B, cha_po_2)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        # (B, targets)

        return x
