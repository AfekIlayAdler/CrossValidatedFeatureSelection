from pathlib import Path

from pandas import read_csv, factorize, Series

from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline
from experiments.run_experiment import run_experiments


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'y'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss'
        , 'hours_per_week', 'native_country', 'y']

    train = read_csv(project_root / "datasets/adult/adult.data", header=None)
    train.columns = columns
    y = Series(factorize(train[y_col_name])[0])
    X = train.drop(columns=[y_col_name])
    for col in X.select_dtypes(include=['O']).columns.tolist():
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    config = Config(
        compute_permutation=True,
        save_results=True,
        one_hot=True,
        contains_num_features=True,
        seed=7,
        predictors=GBM_CLASSIFIERS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline)
    run_experiments(config)
