from pathlib import Path

from numpy import random
from pandas import read_csv, to_datetime, DataFrame, Series

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.run_experiment import run_experiments

"""
dataset from kaggle. contains both categorical and numerical features. ~11k samples
"""


def get_x_y():
    X = DataFrame()
    nrows = 10**3
    for i in range(1,7):
        X[i] = random.randint(0, 2**1, nrows)
    X['continues'] = random.random(nrows)
    y = Series(random.random(nrows))
    return X, y


if __name__ == '__main__':
    config = Config(
        kfold_flag=False,
        compute_permutation=True,
        save_results=True,
        one_hot=False,  # when we add index then it takes to much time..
        contains_num_features=True,
        seed=7,
        kfolds=30,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline)
    run_experiments(config)