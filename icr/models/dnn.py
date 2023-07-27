from typing import Tuple
from torch import nn
from torch.nn import functional as F
from model_utils.config import BaseConfig

class ICRDNNConfig(BaseConfig):

    num_features: int = 56
    num_targets: int = 1

    # channel_base_dim: int = 128
    # hidden_size: int = 2048
    channel_base_dim: int = 64
    hidden_size: Tuple[int, int, int, int] = [150, 125, 100, 750]
    dropout: Tuple[float, float, float, float] = [0.5, 0.35, 0.3, 0.25]


class ICRDNNModel(nn.Module):
    def __init__(self, config: ICRDNNConfig):
        super().__init__()

        # self.hidden_size = [1500, 1250, 1000, 750]
        # self.dropout_value = [0.5, 0.35, 0.3, 0.25]

        self.batch_norm1 = nn.BatchNorm1d(config.num_features)
        self.dense1 = nn.Linear(config.num_features, config.hidden_size[0])

        self.batch_norm2 = nn.BatchNorm1d(config.hidden_size[0])
        self.dropout2 = nn.Dropout(config.dropout[0])
        self.dense2 = nn.Linear(config.hidden_size[0], config.hidden_size[1])

        self.batch_norm3 = nn.BatchNorm1d(config.hidden_size[1])
        self.dropout3 = nn.Dropout(config.dropout[1])
        self.dense3 = nn.Linear(config.hidden_size[1], config.hidden_size[2])

        self.batch_norm4 = nn.BatchNorm1d(config.hidden_size[2])
        self.dropout4 = nn.Dropout(config.dropout[2])
        self.dense4 = nn.Linear(config.hidden_size[2], config.hidden_size[3])

        self.batch_norm5 = nn.BatchNorm1d(config.hidden_size[3])
        self.dropout5 = nn.Dropout(config.dropout[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(config.hidden_size[3], config.num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x
