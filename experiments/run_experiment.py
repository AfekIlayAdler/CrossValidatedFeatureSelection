from numpy import nan
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

from algorithms import LEARNING_RATE
from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE
from experiments.utils import transform_categorical_features


def run_experiment(
        model_name: str, variant: str, get_data: callable, compute_permutation: bool, \
        save_results: bool, contains_num_features: bool, preprocessing_pipeline, models_dict,
        exp_results_path):
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    preprocessing_pipeline.fit(X_train)
    X_train = preprocessing_pipeline.transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    if contains_num_features and model_name == 'ours':
        num_cols = get_num_cols(X.dtypes)
        bin_mapper = BinMapper(max_bins=256, random_state=42)
        X_train.loc[:, num_cols] = bin_mapper.fit_transform(X_train.loc[:, num_cols].values)
        X_test.loc[:, num_cols] = bin_mapper.transform(X_test.loc[:, num_cols].values)

    original_dtypes = X_train.dtypes
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = models_dict[model_name](variant, original_dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                    learning_rate=LEARNING_RATE, subsample=1.)
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    if compute_permutation:
        permutation_train = model.compute_fi_permutation(X_train, y_train).to_dict()
        permutation_test = model.compute_fi_permutation(X_test, y_test).to_dict()
    else:
        empty_dict = Series({col: nan for col in original_dtypes})
        permutation_train = empty_dict
        permutation_test = empty_dict

    results = dict(model=f"{model_name}_{variant}", ntrees=model.get_n_trees(), nleaves=model.get_n_leaves(),
                   error=model.compute_error(y_test, test_prediction), gain=model.compute_fi_gain().to_dict(),
                   permutation_train=permutation_train, permutation_test=permutation_test,
                   shap_train=model.compute_fi_shap(X_train, y_train).to_dict(),
                   shap_test=model.compute_fi_shap(X_test, y_test).to_dict())
    if save_results:
        DataFrame(Series(results)).T.to_csv(exp_results_path)