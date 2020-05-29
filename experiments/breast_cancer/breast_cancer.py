from pandas import Series, DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import RESULTS_DIR, VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, GBM_CLASSIFIERS
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    data = load_breast_cancer()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    return X, y


def worker(model_name, variant):
    exp_name = F"{model_name}_{variant}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    # if exp_results_path.exists():
    #     return
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    if model_name.startswith('ours'):
        num_cols = get_num_cols(X.dtypes)
        bin_mapper = BinMapper(max_bins=256, random_state=42)
        X_train.loc[:, num_cols] = bin_mapper.fit_transform(X_train.loc[:, num_cols].values)
        X_test.loc[:, num_cols] = bin_mapper.transform(X_test.loc[:, num_cols].values)

    results = {'model': F"{model_name}_{variant}"}
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = GBM_CLASSIFIERS[model_name](variant, X.dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
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
    print("************")
    print(model.compute_f1(X_test, y_test))
    print("************")
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    MODELS = {
        'lgbm': ['vanilla'],
        # 'xgboost': ['mean_imputing'],
        # 'catboost': ['vanilla'],
        # 'sklearn': ['mean_imputing'],
        # 'ours_vanilla': ['_'],
        # 'ours_kfold': ['_']
    }

    make_dirs([RESULTS_DIR])
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        make_dirs([RESULTS_DIR / model_name])
        for variant in model_variants:
            worker(model_name, variant)
    print("run took {end - time}")
