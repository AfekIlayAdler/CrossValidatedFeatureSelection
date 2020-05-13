import multiprocessing
from pathlib import Path

from pandas import Series, DataFrame, read_csv, to_datetime
from sklearn.model_selection import train_test_split

from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import RESULTS_DIR, VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, GBM_REGRESSORS
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    def preprocess_df(df):
        df['datetime'] = to_datetime(df['datetime']).dt.hour
        df['datetime'] = df['datetime'].astype('category')
        df['holiday'] = df['holiday'].astype(int)
        df['season'] = df['season'].astype('category')
        df['workingday'] = df['workingday'].astype('category')
        df['weather'] = df['weather'].astype('category')
        X = df[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']]
        y = df['count']
        return X, y
    project_root =  Path(__file__).parent.parent.parent
    train = read_csv(project_root / 'datasets/bike_rental_regression/train.csv')
    X, y = preprocess_df(train)
    return X, y


def worker(model_name, variant, fast):
    exp_name = F"{model_name}_{variant}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    # if exp_results_path.exists():
    #     return
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    # if fast and model_name == 'ours':
    num_cols = get_num_cols(X.dtypes)
    bin_mapper = BinMapper(max_bins=256, random_state=42)
    X_train.loc[:,num_cols]= bin_mapper.fit_transform(X_train.loc[:,num_cols].values)
    X_test.loc[:,num_cols] = bin_mapper.transform(X_test.loc[:,num_cols].values)

    results = {'model': F"{model_name}_{variant}"}
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = GBM_REGRESSORS[model_name](variant, X.dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                       learning_rate=LEARNING_RATE, fast = fast)
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
        'shap_test': model.compute_fi_shap(X_test, y_test).to_dict()
    })
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    FAST = True
    DEBUG = False
    MODELS = {
        'lgbm': ['vanilla'],
        'xgboost': ['one_hot', 'mean_imputing'],
        'catboost': ['vanilla'],
        'sklearn': ['one_hot', 'mean_imputing'],
        'ours': ['Kfold', 'CartVanilla']
    }

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
