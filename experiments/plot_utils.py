import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pandas import DataFrame


def get_regular_paths(one_hot):
    paths = {'Lgbm': "lgbm/lgbm_vanilla",
             'Catboost': "catboost/catboost_vanilla",
             "Ours_kfold": "ours_Kfold/ours_kfold__",
             "Ours_Vanilla": "ours_vanilla/ours_vanilla__",
             "Sklearn_MeanImputing": "sklearn/sklearn_mean_imputing",
             'Xgboost_MeanImputing': "xgboost/xgboost_mean_imputing", }
    if one_hot:
        paths.update({
            "Slearn_OneHot": "sklearn/sklearn_one_hot.csv",
            'Xgboost_OneHot': "xgboost/xgboost_one_hot.csv"})
    return paths


def plot_boxplot(paths, metric):
    l = []
    for model, model_path in paths.items():
        for exp_path in glob.glob(model_path + '*'):
            temp_results_dict = get_yaml_file(exp_path)
            l.append([model, temp_results_dict[metric]])
    plot_df = DataFrame(l, columns=['Models', metric])
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(x="Models", y=metric, data=plot_df)
    ax.set_title(F"{metric}_boxplot")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.figure.savefig(F"{metric}_boxplot.png")


def plot_fi(model_name, df):
    plt.figure()
    plot = df.plot(kind='bar', figsize=(15, 6), title=model_name, ylim=(0, 0.7), rot=45)
    plot.figure.savefig(F"{model_name}_feature_importance.png")


def get_yaml_file(path):
    with open(path) as file:
        temp_fi_dict = yaml.load(file, Loader=yaml.FullLoader)
    return temp_fi_dict


def get_feature_importance(model_name, path):
    fi_cols = ['gain', 'permutation_train', 'permutation_test']
    if not model_name.startswith('Ours'):
        fi_cols += ['shap_train', 'shap_test']
    results = pd.DataFrame()
    all_paths = glob.glob(path + '*')
    n_experiments = len(all_paths)
    for i, path in enumerate(all_paths):
        temp_fi_dict = get_yaml_file(path)
        for fi in fi_cols:
            temp_fi = pd.Series(temp_fi_dict[fi]).sort_index()
            if i == 0:
                results[fi] = temp_fi.values
            else:
                results[fi] += temp_fi.values
    return results / n_experiments


if __name__ == '__main__':
    ONE_HOT = False
    LOO = True
    paths = get_regular_paths(ONE_HOT)
    for k, v in paths.items():
        # results = get_feature_importance(k, v, LOO)
        plot = get_feature_importance(k, v, LOO).plot(kind='bar', figsize=(15, 6), title=k, ylim=(-0.05, 0.6), rot=30)
        plot.figure.savefig(F"{k}_feature_importance.png")
