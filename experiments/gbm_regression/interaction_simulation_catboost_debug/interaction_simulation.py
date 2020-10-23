from pathlib import Path

from numpy import random, zeros
from pandas import DataFrame, Series

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS, GBM_CLASSIFIERS, SEED, KFOLDS, N_EXPERIMENTS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y(a, k1=50,k2 = 200, sigma=20, x1_cat=False):
    X = DataFrame()
    nrows = 10 ** 3
    if x1_cat:
        X['X1'] = random.randint(0, k1, nrows)
        X['X1'] = X['X1'].astype('category')
        X['X2'] = random.randint(0, k2, nrows)
        X['X2'] = X['X2'].astype('category')
        y = a * X['X1'].isin(list(range(k1 // 2))) + (10 - a) * X['X2'].isin(list(range(k2 // 2))) + random.random(
            nrows) * sigma

    else:
        X['X1'] = random.random(nrows)
        X['X2'] = random.randint(0, k1, nrows)
        X['X2'] = X['X2'].astype('category')
        y = a * (X['X1'] > 0.5) + (10 - a) * X['X2'].isin(list(range(k1 // 2))) + random.random(
            nrows) * sigma

    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = True
    KFOLD = False
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    RESULTS_DIR = Path("k_50_sigma_5_x1_num_for_paper_k_200_sigma_20_nrows100/")

    REGRESSION = True
    x, y = get_x_y(2)
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if contains_num_features else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS

    config = Config(
        multiple_experimens=MULTIPLE_EXPERIMENTS,
        n_experiments=10,#100
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
    experiment_configurator(config, True)
