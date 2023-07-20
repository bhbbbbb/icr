# pylint: disable=all
from torch import Tensor
from tqdm import tqdm
from icr.dataset import ICRDataset


def general():
    train_set = ICRDataset('train')

    print('------------- general test -----------------')

    step = 0
    features_cache: Tensor = None
    labels_cache: Tensor = None
    for features, labels in tqdm(train_set.dataloader):
        
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

def subset():

    train_set = ICRDataset('train')

    length = len(train_set)
    train_subset_indices = list(range(int(length * .8)))
    train_subset = train_set.make_subset(train_subset_indices, 'train')
    valid_subset_indices = list(range(length - len(train_subset_indices)))
    valid_subset = train_set.make_subset(valid_subset_indices, 'test')

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
    for features, labels in tqdm(valid_subset.dataloader):
        
        if step == 0:
            features_cache = features
            labels_cache = labels
        
        step += 1
    print(f'features: {features_cache.shape}, {features_cache.dtype}')
    print(f'features: {labels_cache.shape}, {labels_cache.dtype}')
    return

def main():
    general()
    subset()
    

if __name__ == '__main__':
    main()
