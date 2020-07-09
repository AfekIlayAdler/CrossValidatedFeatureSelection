from pathlib import Path

from numpy import random, zeros
from pandas import read_csv, to_datetime, DataFrame, Series

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline, get_preprocessing_pipeline_only_cat
from experiments.experiment_configurator import experiment_configurator

"""
dataset from kaggle. contains both categorical and numerical features. ~11k samples
"""


def get_x_y():
    X = DataFrame()
    nrows = 10**3
    y = Series(zeros(nrows))
    for i in range(1,7):
        X[i] = random.randint(0, 2**i, nrows)
        X[i] = X[i].astype('category')
        y += X[i].isin(list(range(2**(i-1))))*1
        # X[i] = random.random(nrows)
        # y += (X[i] <= 0.5) * 0.5

    X[8] = random.random(nrows)
    y +=  random.random(nrows) + (X[8] <= 0) * 0.5
    # y = Series(random.random(nrows))
    return X, y


if __name__ == '__main__':
    config = Config(
        drop_one_feature_flag = False,
        kfold_flag=False,
        compute_permutation=False,
        save_results=True,
        one_hot=False,  # when we add index then it takes to much time..
        contains_num_features=True, # True
        seed=10,
        kfolds=30,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline) #  get_preprocessing_pipeline
    experiment_configurator(config)