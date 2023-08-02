import os
from scripts.stacking import StackingConfig, train_inference

def main():

    dataset_dir = os.path.realpath('icr-identify-age-related-conditions')
    config = StackingConfig(
        train_csv_path=os.path.join(dataset_dir, 'train.csv'),
        test_csv_path=os.path.join(dataset_dir, 'test.csv'),
        greeks_csv_path=os.path.join(dataset_dir, 'greeks.csv'),
        tabpfn_config={'device': 'cuda:0', 'base_path': dataset_dir},
        # tabpfn_config={'device': 'cuda:0'},
        # tabpfn_config={'device': 'cpu', 'base_path': dataset_dir},
        over_sampling_config=StackingConfig.OverSamplingConfig(
            # sampling_strategy=.5,
            # sampling_strategy={0: 408, 1: 98, 2: 29, 3: 47}, # k=5
            # sampling_strategy={0: 459, 1: 110, 2: 33, 3: 53}, # k = 10
            sampling_strategy={0: 1., 1: 2., 2: 2., 3: 2.}, # k = 10
            method='smote',
        ),
        labels='alpha',
        epsilon_as_feature=True,
        # inner_profiles=[*(f'lgb{i}' for i in range(1, 4)), *(f'xgb{i}' for i in range(1, 4)), 'tab0', 'mtab1'],
        # inner_profiles=['lgb2', 'xgb2', 'tab0', 'mtab1'],
        inner_profiles=['lgb1'],
        # inner_profiles=[*(f'lgb{i}' for i in range(1, 5)), *(f'xgb{i}' for i in range(1, 6)), 'tab0', 'mtab1'],
        stacking_profiles=['lgb1'],
        # stacking_profiles=['lgb1', 'lgb2', 'lgb3', 'lgb4'],
        # passthrough=False,
        passthrough=True,
        prediction_analysis=True,
        n_seeds=5,
        inner_k=10,
        outer_k=10,
    )
    config.display()

    pred = train_inference(config)
    print(pred)
    return

if __name__ == '__main__':
    main()
