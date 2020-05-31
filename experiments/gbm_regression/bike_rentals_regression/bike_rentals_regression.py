from pathlib import Path
from time import time

from numpy import random
from numpy.random import seed

from pandas import Series, DataFrame, read_csv, to_datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import RESULTS_DIR, VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, GBM_REGRESSORS, \
    SUBSAMPLE
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.run_experiment import run_experiments
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    def preprocess_df(df):
        df['datetime'] = to_datetime(df['datetime']).dt.hour
        df['datetime'] = df['datetime'].astype('category')
        df['holiday'] = df['holiday'].astype('category')
        df['season'] = df['season'].astype('category')
        df['workingday'] = df['workingday'].astype('category')
        df['weather'] = df['weather'].astype('category')
        X = df[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']]
        y = df['count']
        # X['index'] = random.randint(0,1000,X.shape[0])
        # X['index'] = X['index'].astype('category')
        return X, y

    project_root = Path(__file__).parent.parent.parent.parent
    train = read_csv(project_root / 'datasets/bike_rental_regression/train.csv')
    X, y = preprocess_df(train)
    return X, y


if __name__ == '__main__':
    config = Config(
        compute_permutation=True,
        save_results=True,
        one_hot=False,
        contains_num_features=True,
        seed=7,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline)
    run_experiments(config)





