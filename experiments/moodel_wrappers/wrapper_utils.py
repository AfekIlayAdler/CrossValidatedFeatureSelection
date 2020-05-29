import shap
from pandas import DataFrame


def normalize_series(s):
    if s.sum() != 0:
        return s / s.sum()
    return s


def get_shap_values(model, x, columns):
    shap_values = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x)
    if type(shap_values) == list:
        shap_values = shap_values[0]
    abs_shap_values = DataFrame(shap_values,columns=columns).apply(abs)
    return abs_shap_values.mean() / abs_shap_values.mean().sum()
