from pathlib import Path

from pandas import Series, DataFrame, read_csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import RESULTS_DIR, VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, GBM_CLASSIFIERS
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    project_root = Path(__file__).parent.parent.parent
    y_col_name = 'click'
    train = read_csv(project_root / 'datasets/criteo_ctr_prediction/train_10000.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


def worker(model_name, variant):
    exp_name = F"{model_name}_{variant}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
        return
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    pipeline = get_preprocessing_pipeline(0.5, cols_to_remove)
    pipeline.fit(X_train)
    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)
    results = {'model': F"{model_name}_{variant}"}
    original_dtypes = X_train.dtypes
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = GBM_CLASSIFIERS[model_name](variant, original_dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                        learning_rate=LEARNING_RATE, subsample=1.)
    model.fit(X_train, y_train)
    results.update({
        'ntrees': model.get_n_trees(),
        'nleaves': model.get_n_leaves(),
        'f1_score': model.compute_f1(X_test, y_test),
        'gain': model.compute_fi_gain().to_dict(),
        'permutation_train': model.compute_fi_permutation(X_train, y_train).to_dict(),
        'permutation_test': model.compute_fi_permutation(X_test, y_test).to_dict(),
        'shap_train': model.compute_fi_shap(X_train, y_train).to_dict(),
        'shap_test': model.compute_fi_shap(X_test, y_test).to_dict()
    })
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    cols_to_remove = ['hour', 'id', 'Unnamed: 0']
    MODELS = {
        'lgbm': ['vanilla'],
        'xgboost': ['mean_imputing'],
        'catboost': ['vanilla'],
        'sklearn': ['mean_imputing'],
        'ours_vanilla': ['_'],
        'ours_kfold': ['_']
    }

    make_dirs([RESULTS_DIR])
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        make_dirs([RESULTS_DIR / model_name])
        for variant in model_variants:
            worker(model_name, variant)
    print("run took {end - time}")
