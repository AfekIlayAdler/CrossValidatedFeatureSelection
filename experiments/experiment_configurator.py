from time import time

from numpy.random import seed
from sklearn.model_selection import train_test_split, KFold

from experiments.default_config import VAL_RATIO, RESULTS_DIR
from experiments.experiment_runner import run_experiment
from experiments.utils import make_dirs


def experiment_configurator(config):
    models = get_variants(config.predictors.is_gbm, config.one_hot)
    start = time()
    make_dirs([RESULTS_DIR])
    for model_name, model_variants in models.items():
        print(f'Working on experiment : {model_name}')
        exp_dir = RESULTS_DIR / model_name
        make_dirs([exp_dir])
        for variant in model_variants:
            config._set_attributes(model_name=model_name, variant=variant)
            seed(config.seed)
            configurations = get_configurations(config, model_name, variant, exp_dir)
            for configuration in configurations:
                if config.exp_results_path.exists():
                    continue
                run_experiment(configuration)

    end = time()
    print(f"run took {end - start} seconds")


def get_configurations(config, model_name, variant, exp_dir):
    # if config.multiple_experimens:
    #     return multiple_experiments_configuration(config, model_name, variant, exp_dir)
    # elif config.kfold_flag and config.drop_one_feature_flag:
    #     return get_kfold_drop_one_feature_configuration(config, model_name, variant, exp_dir)
    # elif config.kfold_flag:
    #     return get_kfold_configuration(config, model_name, variant, exp_dir)
    # elif config.drop_one_feature_flag:
    #     return get_drop_one_feature_configuration(config, model_name, variant, exp_dir)
    # else:
    #     return get_regular_configuration(config, model_name, variant, exp_dir)

    if config.multiple_experimens:
        return multiple_experiments_configuration(config, model_name, variant, exp_dir)
    elif config.kfold_flag:
        return get_kfold_configuration(config, model_name, variant, exp_dir)
    else:
        return get_regular_configuration(config, model_name, variant, exp_dir)


def get_variants(is_gbm: bool, one_hot: bool):
    variants = ['mean_imputing', 'one_hot'] if one_hot else ['mean_imputing']
    if is_gbm:
        return dict(lgbm=['vanilla'], xgboost=variants, catboost=['vanilla'], sklearn=variants,
                    ours_vanilla=['_'], ours_kfold=['_'])
    return dict(sklearn=variants, ours_vanilla=['_'], ours_kfold=['_'])


def multiple_experiments_configuration(config, model_name, variant, exp_dir):
    for i in range(config.n_experiments):
        X, y = config.get_x_y()
        config._set_attributes(
            exp_results_path=exp_dir / F"{model_name}_{variant}_{i}.yaml",
            data=train_test_split(X, y, test_size=VAL_RATIO))
        yield config


def get_kfold_configuration(config, model_name, variant, exp_dir):
    X, y = config.get_x_y()
    kf = KFold(n_splits=config.kfolds, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        config._set_attributes(
            exp_results_path=exp_dir / F"{model_name}_{variant}_{i}.yaml",
            data=(X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]),
            compute_permutation=False)
        yield config


def get_regular_configuration(config, model_name, variant, exp_dir):
    X, y = config.get_x_y()
    config._set_attributes(
        exp_results_path=exp_dir / F"{model_name}_{variant}.yaml",
        data=train_test_split(X, y, test_size=VAL_RATIO))
    yield config

# def get_drop_one_feature_configuration(config, model_name, variant, exp_dir):
#     X, y = config.get_x_y()
#     config._set_attributes(
#         exp_results_path=exp_dir / F"{model_name}_{variant}.yaml",
#         data=train_test_split(X, y, test_size=VAL_RATIO),
#         compute_permutation=True)
#     yield config
#     columns_to_remove = config.columns_to_remove
#     for col in X.columns.tolist():
#         config._set_attributes(
#             exp_results_path=exp_dir / F"{model_name}_{variant}_{col}.yaml",
#             data=train_test_split(X, y, test_size=VAL_RATIO),
#             compute_permutation=False,
#             columns_to_remove= columns_to_remove + [col])
#         yield config


# def get_kfold_drop_one_feature_configuration(config, model_name, variant, exp_dir):
#     X, y = config.get_x_y()
#     kf = KFold(n_splits=config.kfolds, shuffle=True)
#     for i, (train_index, test_index) in enumerate(kf.split(X)):
#         data = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
#         config._set_attributes(
#             exp_results_path=exp_dir / F"{model_name}_{variant}_{i}.yaml",
#             data=data,
#             compute_permutation=True)
#         yield config
#
#     columns_to_remove = config.columns_to_remove
#     for i, (train_index, test_index) in enumerate(kf.split(X)):
#         data = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
#         for col in X.columns.tolist():
#             config._set_attributes(
#                 exp_results_path=exp_dir / F"{model_name}_{variant}_{col}_{i}.yaml",
#                 data=data,
#                 compute_permutation=False,
#                 columns_to_remove= columns_to_remove + [col])
#             yield config
