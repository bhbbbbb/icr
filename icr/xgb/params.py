xgb_params0 = {
    'learning_rate': 0.413327571405248,
    'booster': 'gbtree',
    'lambda': 0.0000263894617720096,
    'alpha': 0.000463768723479341,
    'subsample': 0.237467672874133,
    'colsample_bytree': 0.618829300507829,
    'max_depth': 5,
    'min_child_weight': 9,
    'eta': 2.09477807126539E-06,
    'gamma': 0.000847289463422307,
    'grow_policy': 'depthwise',
    'n_jobs': -1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    # 'tree_method': 'gpu_hist',
    # 'predictor': 'gpu_predictor',
}
xgb_params1 = {
    'n_estimators': 900,
    'learning_rate': 0.09641232707445854,
    'booster': 'gbtree',
    'lambda': 4.666002223704784,
    'alpha': 3.708175990751336,
    'subsample': 0.6100174145229473,
    'colsample_bytree': 0.5506821152321051,
    'max_depth': 7,
    'min_child_weight': 3,
    'eta': 1.740374368661041,
    'gamma': 0.007427363662926455,
    'grow_policy': 'depthwise',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    # 'tree_method': 'gpu_hist',
    # 'predictor': 'gpu_predictor',
}

xgb_params2 = {
    'n_estimators': 650,
    'learning_rate': 0.012208383405206188,
    'booster': 'gbtree',
    'lambda': 0.009968756668882757,
    'alpha': 0.02666266827121168,
    'subsample': 0.7097814108897231,
    'colsample_bytree': 0.7946945784285216,
    'max_depth': 3,
    'min_child_weight': 4,
    'eta': 0.5480204506554545,
    'gamma': 0.8788654128774149,
    'scale_pos_weight': 4.71,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    # 'tree_method': 'gpu_hist',
    # 'predictor': 'gpu_predictor',
}

xgb_params3 = {
    'colsample_bytree': 0.5646751146007976,
    'gamma': 7.788727238356553e-06,
    'learning_rate': 0.1419865761603358,
    'max_bin': 824,
    'min_child_weight': 1,
    'reg_alpha': 1.6259583347890365e-07,
    'reg_lambda': 2.110691851528507e-08,
    'subsample': 0.879020578464637,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'n_jobs': -1,
    'verbosity': 0,
    # 'tree_method': 'gpu_hist',
    # 'predictor': 'gpu_predictor',
}

xgb_params4 = {
    'colsample_bytree': 0.4836462317215041,
    'eta': 0.05976752607337169,
    'gamma': 1,
    'lambda': 0.2976432557733288,
    'max_depth': 6,
    'min_child_weight': 1,
    'n_estimators': 550,
    'objective': 'binary:logistic',
    'scale_pos_weight': 4.260162886376033,
    'subsample': 0.7119282378433924,
}

xgb_params5 = {
    'colsample_bytree': 0.8757972257439255,
    'gamma': 0.11135738771999848,
    'max_depth': 7,
    'min_child_weight': 3,
    'reg_alpha': 0.4833998914998038,
    'reg_lambda': 0.006223568555619563,
    'scale_pos_weight': 8,
    'subsample': 0.7056434340275685,
}

xgb_params6 = {
    'max_depth': 5, 
    'min_child_weight': 2.934487833919741,
    'learning_rate': 0.11341944575807082, 
    'subsample': 0.9045063514419968,
    'gamma': 0.4329153382843715,
    'colsample_bytree': 0.38872702868412506,
    'colsample_bylevel': 0.8321880031718571,
    'colsample_bynode': 0.802355707802605,
}

profiles_list = [
    xgb_params0,
    xgb_params1,
    xgb_params2,
    xgb_params3,
    xgb_params4,
    xgb_params5,
    xgb_params6,
]
multi_profile_list = [
    {**profile, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}\
        for profile in profiles_list
]
profiles = {
    **{ f'xgb{idx}': params for idx, params in enumerate(profiles_list) },
    **{ f'mxgb{idx}': params for idx, params in enumerate(multi_profile_list) },
}
