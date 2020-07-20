from pathlib import Path

from pandas import read_csv

from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS, GBM_REGRESSORS, N_EXPERIMENTS, SEED, KFOLDS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline
from experiments.experiment_configurator import experiment_configurator


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'click'
    train = read_csv(project_root / 'datasets/criteo_ctr_prediction/train_10000.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name, 'hour', 'id', 'Unnamed: 0'])
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = False
    KFOLD = True
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    CONTAINS_NUM_FEATURES = False

    REGRESSION = False
    pp = get_preprocessing_pipeline if CONTAINS_NUM_FEATURES else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS
    config = Config(
        multiple_experimens=MULTIPLE_EXPERIMENTS,
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
