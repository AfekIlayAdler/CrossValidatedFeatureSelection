from numpy import random, zeros
from pandas import DataFrame, Series

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS, GBM_CLASSIFIERS, SEED, KFOLDS, N_EXPERIMENTS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y():
    X = DataFrame()
    nrows = 10 ** 3
    y = Series(zeros(nrows))
    for i in range(1, 7):
        X[i] = random.randint(0, 2 ** i, nrows)
        X[i] = X[i].astype('category')
        y += X[i].isin(list(range(2 ** (i - 1)))) * 1

    y += random.random(nrows)
    y = Series(random.random(nrows))
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENS = True
    KFOLD = False
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    CONTAINS_NUM_FEATURES = False

    REGRESSION = True
    pp = get_preprocessing_pipeline if CONTAINS_NUM_FEATURES else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS

    config = Config(
        multiple_experimens=MULTIPLE_EXPERIMENS,
        n_experiments=N_EXPERIMENTS,
        kfold_flag=KFOLD,
        compute_permutation=COMPUTE_PERMUTATION,
        save_results=True,
        one_hot=ONE_HOT,
        contains_num_features=CONTAINS_NUM_FEATURES,
        seed=SEED,
        kfolds=KFOLDS,
        predictors=predictors,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=pp)
    experiment_configurator(config)
