# pylint: disable=all
import sys
from torch import Tensor
from tqdm import tqdm
from icr.dataset import ICRDataset, ICRDatasetConfig


def general(config):
    train_set = ICRDataset('train', config)

    print('------------- general test -----------------')

    step = 0
    features_cache: Tensor = None
    labels_cache: Tensor = None
    dataloader = train_set.dataloader

    for features, labels in tqdm(dataloader):
        
        if step == 0:
            features_cache = features
            labels_cache = labels
        
        step += 1

    print(f'features: {features_cache.shape}, {features_cache.dtype}')
    print(f'features: {labels_cache.shape}, {labels_cache.dtype}')
    assert features_cache.shape[1] == 56
    assert labels_cache.shape[0] == features_cache.shape[0]
    assert len(labels_cache.shape) == 1
    assert features_cache.is_floating_point()
    assert not labels_cache.is_floating_point()
    print('-----------------------------')
    return

def subset(config):

    train_set = ICRDataset('train', config) # <<<<<<<<<<<<<

    length = len(train_set)
    train_subset_indices = list(range(int(length * .8)))
    train_subset = train_set.make_subset(train_subset_indices, 'train')
    valid_subset_indices = list(range(int(length * .8), length))
    valid_subset = train_set.make_subset(valid_subset_indices, 'valid')

    print('train_subset:')
    step = 0
    features_cache = None
    labels_cache = None
    for features, labels in tqdm(train_subset.dataloader):
        
        if step == 0:
            features_cache = features
            labels_cache = labels
        
        step += 1

    print(f'features: {features_cache.shape}, {features_cache.dtype}')
    print(f'features: {labels_cache.shape}, {labels_cache.dtype}')
    print('-----------------------------')
    print('valid_subset')
    step = 0
    features_cache = None
    labels_cache = None
    for features in tqdm(valid_subset.dataloader):
        
        if step == 0:
            labels_cache = labels
        
        step += 1
    print(f'features: {labels_cache.shape}, {labels_cache.dtype}')
    return

def main():
    
    import os
    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    sys.path.append(dataset_dir)
    config = ICRDatasetConfig(
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        batch_size_train=64,
        batch_size_eval=128,
        num_workers=0,
    )
    general(config)
    subset(config)
    

if __name__ == '__main__':
    main()
