from pandas import DataFrame, Series
from sklearn.datasets import load_boston

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_num
from experiments.run_experiment import run_experiments


def get_x_y():
    data = load_boston()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    # X['index'] = random.randint(0,500,X.shape[0])
    # X['index'] = X['index'].astype('category')
    return X, y


if __name__ == '__main__':
    config = Config(
        compute_permutation=True,
        save_results=True,
        one_hot=False,  # takes to much time
        contains_num_features=True,
        seed=7,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline_only_num)
    run_experiments(config)
