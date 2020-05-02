import shap
from pandas import DataFrame


def normalize_series(s):
    if s.sum() != 0:
        return s / s.sum()
    return s


def get_shap_values(model, x, columns):
    abs_shap_values = DataFrame(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x),
                                columns=columns).apply(abs)
    return abs_shap_values.mean() / abs_shap_values.mean().sum()
