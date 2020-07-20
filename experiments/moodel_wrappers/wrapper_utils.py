import shap
from numpy.random import permutation
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, log_loss


def permute_col(df: DataFrame, col: str):
    col_dtype = df[col].dtype
    df[col] = permutation(df[col].values)
    df[col] = df[col].astype(col_dtype)
    col_doesnt_contains_nan = not df[col].isna().any()
    assert col_doesnt_contains_nan, "permutated_x contain nan values"


def normalize_series(s):
    s = s.apply(lambda x: max(x, 0))
    if s.sum() != 0:
        return s / s.sum()
    return s


def get_shap_values(model, x, columns):
    shap_values = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x)
    if type(shap_values) == list:
        shap_values = shap_values[0]
    abs_shap_values = DataFrame(shap_values, columns=columns).apply(abs)
    return abs_shap_values.mean() / abs_shap_values.mean().sum()


def regression_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def classification_error(y_true, y_pred):
    return log_loss(y_true, y_pred)
