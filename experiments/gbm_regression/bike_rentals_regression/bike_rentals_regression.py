from pathlib import Path

from numpy import random
from pandas import read_csv, to_datetime

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.run_experiment import run_experiments

"""
dataset from kaggle. contains both categorical and numerical features. ~11k samples
"""


def get_x_y():
    def preprocess_df(df):
        df['datetime'] = to_datetime(df['datetime']).dt.hour
        for col in ['datetime', 'holiday', 'season', 'workingday', 'weather']:
            df[col] = df[col].astype('category')
        X = df[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']]
        y = df['count']
        ######
        X['index'] = random.random()
        # X['index'] = random.randint(0, 500, X.shape[0])
        # X['index'] = X['index'].astype('category')
        # columns = ['index', 'temp', 'atemp', 'weather', 'humidity']
        # X = X[columns]
        return X, y

    project_root = Path(__file__).parent.parent.parent.parent
    train = read_csv(project_root / 'datasets/bike_rental_regression/train.csv')
    X, y = preprocess_df(train)
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
