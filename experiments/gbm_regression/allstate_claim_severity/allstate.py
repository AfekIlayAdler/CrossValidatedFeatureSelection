from pathlib import Path

from pandas import read_csv

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat
from experiments.experiment_configurator import experiment_configurator


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'loss'
    train = read_csv(project_root / 'datasets/allstate_claim_severity/train.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    cols = [f"cat{i}" for i in range(80, 117)]
    X = X[cols]
    return X, y


if __name__ == '__main__':
    config = Config(
        compute_permutation=True,
        save_results=True,
        one_hot=False,  # takes to much time
        contains_num_features=False,
        seed=7,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline_only_cat)
    experiment_configurator(config)


