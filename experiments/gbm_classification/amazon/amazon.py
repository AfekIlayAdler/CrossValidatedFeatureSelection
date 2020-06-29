from pathlib import Path

from pandas import read_csv

from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat
from experiments.run_experiment import run_experiments


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'ACTION'
    train = read_csv(project_root / 'datasets/amazon_from_catboost_paper/train.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name, 'ROLE_FAMILY_DESC'])
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    config = Config(
        kfold_flag = True,
        compute_permutation=False,
        save_results=True,
        one_hot=False,  # takes to much time
        contains_num_features=False,
        seed=7,
        kfolds = 30,
        predictors=GBM_CLASSIFIERS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline_only_cat)
    run_experiments(config)
