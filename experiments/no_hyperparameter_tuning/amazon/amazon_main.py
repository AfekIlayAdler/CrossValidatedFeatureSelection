from pathlib import Path

from numpy import nan, unique
from pandas import read_csv, Series, DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from algorithms import FastCartGradientBoostingClassifierKfold, FastCartGradientBoostingClassifier
from experiments.default_config import VAL_RATIO


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'ACTION'
    train = read_csv(project_root / 'datasets/amazon_from_catboost_paper/train.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


def run_experiment(
        model_name: str, get_data: callable, compute_permutation: bool, \
        save_results: bool, model,exp_results_path):
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    original_dtypes = X_train.dtypes
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    if compute_permutation:
        permutation_train = model.compute_fi_permutation(X_train, y_train).to_dict()
        permutation_test = model.compute_fi_permutation(X_test, y_test).to_dict()
    else:
        empty_dict = Series({col: nan for col in original_dtypes})
        permutation_train = empty_dict
        permutation_test = empty_dict

    is_classification = len(unique(y)) == 2
    if is_classification:
        logloss = log_loss(y_test, test_prediction)
    else:
        logloss = nan
    results = dict(model=f"{model_name}", ntrees=model.get_n_trees(), nleaves=model.get_n_leaves(),
                   error=model.compute_error(y_test, test_prediction), logloss=logloss,
                   gain=model.compute_fi_gain().to_dict(),
                   permutation_train=permutation_train, permutation_test=permutation_test,
                   shap_train=model.compute_fi_shap(X_train, y_train).to_dict(),
                   shap_test=model.compute_fi_shap(X_test, y_test).to_dict())

    if save_results:
        DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    models = {
        'kfold_no_hyper': FastCartGradientBoostingClassifierKfold(max_depth=100,
                                                                  n_estimators=1000,
                                                                  learning_rate=0.1),

        'kfold_height': FastCartGradientBoostingClassifierKfold(max_depth=3,
                                                                n_estimators=1000,
                                                                learning_rate=0.1),
        'vanilla_no_hyper': FastCartGradientBoostingClassifier(max_depth=100,
                                                               n_estimators=1000,
                                                               learning_rate=0.1),

        'vanilla_height': FastCartGradientBoostingClassifier(max_depth=3,
                                                             n_estimators=1000,
                                                             learning_rate=0.1)}

    for exp_name, model in models.items():
        print(f"run exp {exp_name}")
        run_experiment(exp_name,get_x_y, False, True, model, f"{exp_name}.csv" )
