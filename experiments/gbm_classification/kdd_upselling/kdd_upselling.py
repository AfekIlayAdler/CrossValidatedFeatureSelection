from pathlib import Path

from pandas import read_csv, factorize, Series

from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS
from experiments.default_config import GBM_REGRESSORS, N_EXPERIMENTS, SEED, KFOLDS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'y'
    train = read_csv(project_root / "datasets/kdd_upselling/train.csv")
    y = Series(factorize(train[y_col_name])[0])
    X = train.drop(columns=[y_col_name])
    print(X.shape)
    for col in X.select_dtypes(include=['O']).columns.tolist():
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = False
    KFOLD = False
    ONE_HOT = False
    COMPUTE_PERMUTATION = False
    CONTAINS_NUM_FEATURES = True

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