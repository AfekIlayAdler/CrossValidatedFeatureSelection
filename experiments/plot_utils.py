import glob
import warnings

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


def plot_metrics(paths, metric):
    l = []
    for model, model_path in paths.items():
        for exp_path in glob.glob(model_path + '*'):
            temp_results_dict = get_yaml_file(exp_path)
            l.append([model, temp_results_dict[metric]])
    plot_df = DataFrame(l, columns=['Models', metric])
    is_barplot = (plot_df['Models'].value_counts() == 1).any()
    plt.figure(figsize=(15, 8))
    if is_barplot:
        ax = sns.barplot(x="Models", y=metric, data=plot_df)
        ax.set_title(F"{metric}_barplot")
    else:
        ax = sns.boxplot(x="Models", y=metric, data=plot_df)
        ax.set_title(F"{metric}_boxplot")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.figure.savefig(F"{metric}.png")


def plot_fi(model_name, df):
    plot = df.plot(kind='bar', figsize=(15, 6), title=model_name, ylim=(0, 0.7), rot=45)
    plot.figure.savefig(F"{model_name}_feature_importance.png")



def get_yaml_file(path):
    with open(path) as file:
        temp_fi_dict = yaml.load(file, Loader=yaml.FullLoader)
    return temp_fi_dict


def get_feature_importance(model_name, path, with_permutation=True):
    fi_cols = ['gain']
    if with_permutation:
        fi_cols += ['permutation_train', 'permutation_test']
    if not model_name.startswith('Ours'):
        fi_cols += ['shap_train', 'shap_test']
    all_paths = glob.glob(path + '*')
    fi_counter = {col : 0 for col in fi_cols}
    for i, path in enumerate(all_paths):
        temp_fi_dict = get_yaml_file(path)
        for j, fi in enumerate(fi_cols):
            temp_fi = pd.Series(temp_fi_dict[fi])
            # temp_fi.index = [int(col) for col in temp_fi.index]
            temp_fi = temp_fi.sort_index()
            contains_nan = temp_fi.isna().any()
            if contains_nan:
                message = F"{path} feature importance {fi} is nan"
                warnings.warn(message)
            else:
                if i == 0:
                    if j == 0:
                        results = pd.DataFrame(index=temp_fi.index)
                    # print(path, fi)
                    results[fi] = temp_fi.values
                else:
                    results[fi] += temp_fi.values
                fi_counter[fi] += 1
    for col in fi_cols:
        results[col] /= fi_counter[col]
    return results



