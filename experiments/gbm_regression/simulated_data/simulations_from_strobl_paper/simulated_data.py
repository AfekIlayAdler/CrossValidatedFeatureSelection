from pathlib import Path

from numpy import random, zeros
from pandas import DataFrame, Series

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS, GBM_CLASSIFIERS, SEED, KFOLDS, N_EXPERIMENTS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y():
    y_is_noise = False
    num_feature = False
    num_feature_connected = False
    connected_range = {5}

    X = DataFrame()
    nrows = 10 ** 4
    y = Series(zeros(nrows))
    for i in range(5, 10):
        X[i] = random.randint(0, 2 ** i, nrows)
        X[i] = X[i].astype('category')
        if i in connected_range:
            y += X[i].isin(list(range(2 ** (i - 2)))) * 1

    X.columns = [str(i) for i in X.columns]
    if num_feature:
        X['numeric'] = random.random(nrows) * 1
        if num_feature_connected:
            y += (X['numeric'] <= 0) * 1

    y += random.random(nrows) * 3
    if y_is_noise:
        y = Series(random.random(nrows))
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = True
    KFOLD = False
    ONE_HOT = True
    COMPUTE_PERMUTATION = True
    RESULTS_DIR = Path("5_is_connected_for_boxplot/")

    REGRESSION = True
    x, y = get_x_y()
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if contains_num_features else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS

    config = Config(
        multiple_experimens=MULTIPLE_EXPERIMENTS,
        n_experiments=100,
        kfold_flag=KFOLD,
        compute_permutation=COMPUTE_PERMUTATION,
        save_results=True,
        one_hot=ONE_HOT,
        contains_num_features=contains_num_features,
        seed=SEED,
        kfolds=KFOLDS,
        predictors=predictors,
        columns_to_remove=[],
        get_x_y=get_x_y,
        results_dir=RESULTS_DIR,
        preprocessing_pipeline=pp)
    experiment_configurator(config)
