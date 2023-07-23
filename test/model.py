import torch
from torch import Tensor
from icr.models.icr_model import ICRModel, ICRModelConfig
from icr.model_utils.icr_model_utils import ICRModelUtils, ICRModelUtilsConfig
from icr.dataset import ICRDataset

# class Config(ICRModelUtilsConfig, ICRModelConfig)
device = 'cuda:0'
config = ICRModelConfig(
    num_features=56,
    num_targets=1,
    # channel_base_dim=,
)
model = ICRModel(config)
model.to(device)

def main():
    train_set = ICRDataset('train') # <<<<<<<<<<<<

    print('------------- general test -----------------')

    step = 0
    for features, labels in train_set.dataloader:
        
        features = features.to(torch.float).to(device)
        labels = labels.to(torch.long).to(device)

        # with torch.autocast('cuda'):
        predictions: Tensor = model.forward(features)
            # predictions = model.forward(features)
        print(predictions)
        print(predictions.shape)
        input('continue...')
        step += 1


if __name__ == '__main__':
    main()
