from pathlib import Path
from time import time

from numpy.random import seed
from pandas import read_csv

from experiments.default_config import RESULTS_DIR, GBM_CLASSIFIERS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat
from experiments.run_experiment import run_experiment
from experiments.utils import make_dirs


def get_x_y():
    project_root = Path(__file__).parent.parent.parent
    y_col_name = 'click'
    train = read_csv(project_root / 'datasets/criteo_ctr_prediction/train_10000.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    SEED = 7
    MODELS = dict(lgbm=['vanilla'], xgboost=['mean_imputing'], catboost=['vanilla'], sklearn=['mean_imputing'],
                  ours_vanilla=['_'], ours_kfold=['_'])
    seed(SEED)
    start = time()
    make_dirs([RESULTS_DIR])
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        exp_dir = RESULTS_DIR / model_name
        make_dirs([exp_dir])
        for variant in model_variants:
            exp_name = F"{model_name}_{variant}.csv"
            exp_results_path = exp_dir / exp_name
            if exp_results_path.exists():
                continue
            run_experiment(
                model_name=model_name,
                variant=variant,
                get_data=get_x_y,
                compute_permutation=True,
                save_results=True,
                contains_num_features=False,
                preprocessing_pipeline=get_preprocessing_pipeline_only_cat(0.5, ['hour', 'id', 'Unnamed: 0']),
                models_dict=GBM_CLASSIFIERS,
                exp_results_path=exp_results_path)

    end = time()
    print(f"run took {end - start} seconds")
