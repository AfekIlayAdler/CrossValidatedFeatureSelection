import multiprocessing

from numpy import arange
from pandas import Series, DataFrame
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from experiments.default_config import RESULTS_DIR, VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, GBM_REGRESSORS
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    data = load_boston()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    """
    add cat index
    """
    # X['index'] = arange(X.shape[0])
    # X['index'] = X['index'].astype('category')
    return X, y


def worker(model_name, variant, fast):
    exp_name = F"{model_name}_{variant}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
       return
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    results = {'model': F"{model_name}_{variant}"}

    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = GBM_REGRESSORS[model_name](variant, X.dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                       learning_rate=LEARNING_RATE, fast=fast)
    model.fit(X_train, y_train)
    print("finished fittin the model")
    results.update({
        'ntrees': model.get_n_trees(),
        'nleaves': model.get_n_leaves(),
        'rmse': model.compute_rmse(X_test, y_test),
        'gain': model.compute_fi_gain().to_dict(),
        'permutation_train': model.compute_fi_permutation(X_train, y_train).to_dict(),
        'permutation_test': model.compute_fi_permutation(X_test, y_test).to_dict(),
        'shap_train': model.compute_fi_shap(X_train, y_train).to_dict(),
        'shap_test': model.compute_fi_shap(X_test, y_test).to_dict()})
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    FAST = True
    DEBUG = True
    MODELS = {
        # 'lgbm': ['vanilla'],
        # 'xgboost': ['mean_imputing'],
        # 'catboost': ['vanilla'],
        # 'sklearn': ['mean_imputing'],
        'ours': ['Kfold', 'CartVanilla']}

    make_dirs([RESULTS_DIR])
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        args = []
        make_dirs([RESULTS_DIR / model_name])
        for variant in model_variants:
            if DEBUG:
                worker(model_name, variant, FAST)
            else:
                args.append((model_name, variant, FAST))
        if not DEBUG:
            with multiprocessing.Pool() as process_pool:
                process_pool.starmap(worker, args)
