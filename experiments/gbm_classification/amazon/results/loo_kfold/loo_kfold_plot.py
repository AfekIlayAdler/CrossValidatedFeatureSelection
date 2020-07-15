import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from pathlib import Path

from experiments.moodel_wrappers.wrapper_utils import normalize_series


FOLDS  = 5

def get_regular_paths(one_hot):
    paths = {
            # 'Lgbm': "lgbm/lgbm_vanilla.csv",
             'Catboost': "catboost/catboost_vanilla.csv",
             "Ours_kfold": "ours_Kfold/ours_kfold__.csv",
             "Ours_Vanilla": "ours_vanilla/ours_vanilla__.csv",
             "Sklearn_MeanImputing": "sklearn/sklearn_mean_imputing.csv",
             'Xgboost_MeanImputing': "xgboost/xgboost_mean_imputing.csv", }
    if one_hot:
        paths.update({
            "Slearn_OneHot": "sklearn/sklearn_one_hot.csv",
            'Xgboost_OneHot': "xgboost/xgboost_one_hot.csv"})
    return paths


def get_loo_fi(path, features):
    fi = []
    all_features__metric = pd.read_csv(path)['logloss'][0]
    start_string, end_string = path.split('.')
    for feature in features:
        temp_path = F"{start_string}_{feature}.{end_string}"
        sub_group_metric = pd.read_csv(temp_path)['logloss'][0]
        fi.append(sub_group_metric - all_features__metric)
    return fi


def get_fi(path, cols, converters=None):
    start_string, end_string = path.split('.')
    dataframes = []
    for i in range(FOLDS):
        temp_df = pd.DataFrame()
        temp_path = F"{start_string}_{i}.{end_string}"
        df = pd.read_csv(temp_path, converters=converters)
        for col in cols:
            temp_df[col] = pd.Series(df.loc[0, col])
        print("ROLE_CODE" in temp_df.index.tolist())
        print(temp_df.shape)
        temp_df = temp_df.sort_index()
        dataframes.append(temp_df)

    results = dataframes[0]
    for i in range(1, FOLDS):
        results += dataframes[i]
    return results / FOLDS


def get_feature_importance(exp, path, loo):
    all_cols = ['gain', 'permutation_train', 'permutation_test', 'shap_train', 'shap_test']
    our_cols = ['gain', 'permutation_train', 'permutation_test']
    cols = our_cols if exp.startswith('Ours') else all_cols
    fi_df = get_fi(path, cols, {col: literal_eval for col in cols})
    if loo:
        fi_df['loo'] = get_loo_fi(path, fi_df.index.tolist())
        fi_df['loo'] = normalize_series(fi_df['loo'])
    return fi_df


if __name__ == '__main__':
    ONE_HOT = False
    LOO = True
    paths = get_regular_paths(ONE_HOT)
    for k, v in paths.items():
        # results = get_feature_importance(k, v, LOO)
        plot = get_feature_importance(k, v, LOO).plot(kind='bar', figsize=(15, 6), title=k, ylim=(-0.05, 0.6), rot=45)
        plot.figure.savefig(F"{k}_feature_importance.png")
