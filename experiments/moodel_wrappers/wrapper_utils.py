import shap
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, f1_score


def normalize_series(s):
    if s.sum() != 0:
        s = s.apply(lambda x: max(x,0))
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
    f1 = f1_score(y_true, (y_pred > 0.5) * 1)
    return 1 - f1
