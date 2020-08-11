import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pandas import DataFrame, Series


def plot_fi(plot_type, folder, save, one_hot=False):
    if plot_type == 'BARPLOT':
        normalize = True
        plot_func = plot_barplot
    else:
        normalize = False
        plot_func = plot_boxplot

    paths = get_regular_paths(one_hot=one_hot)
    metrics = []

    for model, model_path in paths.items():
        try:
            temp_fi, temp_metrics = get_data(model, F"{folder}/{model_path}", normalize)
            metrics.append(metrics_to_df(temp_metrics, model))
            save_path = folder if save else None
            plot_func(temp_fi, model, (15, 5), save_path)
        except:
            continue

    metrics = pd.concat(metrics)
    metrics.to_csv(F"{folder}/metrics_summary.csv", index=None)
    for col in metrics.columns[:-1]:
        plt.figure(figsize=(15, 5))
        ax = sns.boxplot(x="model", y=col, data=metrics)
        ax.set_title(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.show()
        if save_path is not None:
            ax.get_figure().savefig(f"{save_path}/{col}.png")


def metrics_to_df(metrics, model):
    results = pd.DataFrame()
    for k, v in metrics.to_dict().items():
        results[k] = v
    results['model'] = model
    return results


def get_regular_paths(one_hot):
    paths = {'LGBM': "lgbm/lgbm_vanilla",
             'CATBOOST': "catboost/catboost_vanilla",
             "Ours": "ours_Kfold/ours_kfold__",
             "Vanilla_GBM": "ours_vanilla/ours_vanilla__",
             "SKLEARN_MI": "sklearn/sklearn_mean_imputing",
             'XGBOOST_MI': "xgboost/xgboost_mean_imputing", }
    if one_hot:
        paths.update({
            "SKLEARN_OH": "sklearn/sklearn_one_hot",
            'XGBOOST_OH': "xgboost/xgboost_one_hot"})
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


def plot_boxplot(fi, title, figsize, save_path, one_plot=False):
    if one_plot:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(x="index", y="fi_value", hue="fi_type", data=fi)
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.show()
        if save_path is not None:
            ax.get_figure().savefig(f"{save_path}/{title}.png")
    else:
        for temp_fi in fi.fi_type.unique():
            plt.figure(figsize=figsize)
            sns.boxplot(x="index", y="fi_value", data=fi[fi['fi_type'] == temp_fi]).set_title(F"{title}_{temp_fi}")
            plt.show()
            if title in ['CATBOOST', 'Ours']:
                plt.figure(figsize=figsize)
                sns.boxplot(x="index", y="fi_value",
                            data=fi[(fi['fi_type'] == temp_fi) & (fi['index'] != '5')]).set_title(F"{title}_{temp_fi}")
                plt.show()
            if save_path is not None:
                sns.plt.savefig(f"{save_path}/{title}_{temp_fi}.png")


def plot_barplot(fi, title, figsize, save_path):
    plt.figure(figsize=figsize)
    ax = sns.barplot(x="index", y="fi_value", hue="fi_type", data=fi)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
    if save_path is not None:
        ax.get_figure().savefig(f"{save_path}/{title}.png")


def get_yaml_file(path):
    with open(path) as file:
        temp_fi_dict = yaml.load(file, Loader=yaml.FullLoader)
    return temp_fi_dict


def get_data(model_name, path, normalize, with_permutation=True):
    fi_cols = ['gain']
    if with_permutation:
        fi_cols += ['permutation_train', 'permutation_test']
    if not model_name.startswith('Ours'):
        fi_cols += ['shap_train', 'shap_test']
    all_paths = glob.glob(path + '*')
    fi_dataframes, metrics_series = [], []
    for i, path in enumerate(all_paths):
        data = get_yaml_file(path)
        metrics = Series({k: v for k, v in data.items() if k in ['nleaves', 'ntrees', 'error']})
        fi = DataFrame({k: v for k, v in data.items() if k in fi_cols})
        if normalize:
            fi = fi.clip(lower=0)
            fi = fi.divide(fi.sum(), axis=1)
        fi = fi.reset_index().melt(id_vars='index', var_name='fi_type', value_name='fi_value')
        fi['exp_number'] = i
        fi_dataframes.append(fi)
        metrics_series.append(metrics)
    final_fi_df = pd.concat(fi_dataframes)
    final_metrics_df = pd.concat(metrics_series).to_frame().reset_index().groupby('index')[0].apply(list)
    final_metrics_df['model'] = model_name
    final_fi_df['model'] = model_name
    return final_fi_df, final_metrics_df


def plot_fi_interaction(folder, one_hot=False):
    normalize = True
    paths = get_regular_paths(one_hot=one_hot)
    metrics, feature_importances = [], []

    for model, model_path in paths.items():
        for a in range(11):
            temp_fi, temp_metrics = get_data(model, F"{folder}/{model_path}_{a}_", normalize)
            temp_metrics = metrics_to_df(temp_metrics, model)
            temp_fi['a'] = a
            temp_metrics['a'] = a
            temp_fi['model'] = model
            metrics.append(temp_metrics)
            feature_importances.append(temp_fi)

    metrics = pd.concat(metrics)
    feature_importances = pd.concat(feature_importances)
    x2_feature_importances = feature_importances[(feature_importances['index'] == 'X2')]
    for fi in x2_feature_importances.fi_type.unique():
        plot_data = x2_feature_importances[(x2_feature_importances.fi_type == fi)]
        plt.figure(figsize=(15, 15))
        plot_data = plot_data.groupby(['model', 'a'])['fi_value'].mean().to_frame().reset_index()
        ax = sns.lineplot(x="a", y="fi_value", hue="model", data=plot_data)
        ax.axhline(0, ls='--')
        ax.set_title(fi)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.show()

    plot_data = x2_feature_importances[(x2_feature_importances['a'] == 10)].groupby(['model', 'fi_type'])[
        'fi_value'].mean().to_frame().reset_index()
    plt.figure(figsize=(15, 15))
    sns.barplot(x="model", y="fi_value", hue="fi_type", data=plot_data)
    plt.show()

    plt.figure(figsize=(15, 15))
    sns.barplot(x="model", y="fi_value", data=plot_data[plot_data.fi_type == 'permutation_test'])
    plt.show()

    # metrics.to_csv(F"{folder}/metrics_summary.csv", index=None)
    plt.figure(figsize=(15, 10))
    ax = sns.lineplot(x="a", y="error", hue="model", err_style='bars', data=metrics)
    ax.set_title('Error')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()


def get_results_data(folder, normalize=False, one_hot=False):
    paths = get_regular_paths(one_hot=one_hot)
    metrics, feature_importances = [], []

    for model, model_path in paths.items():
        temp_fi, temp_metrics = get_data(model, F"{folder}/{model_path}", normalize)
        temp_metrics = metrics_to_df(temp_metrics, model)
        metrics.append(temp_metrics)
        feature_importances.append(temp_fi)
        metrics.append(temp_metrics)
        feature_importances.append(temp_fi)

    metrics = pd.concat(metrics)
    feature_importances = pd.concat(feature_importances)
    return feature_importances, metrics


if __name__ == '__main__':
    import matplotlib

    # matplotlib.use('Agg')  # if we dont want to see the plots
    folder = "C:/Users/afeki/Desktop/code/CrossValidatedFeatureSelection/experiments/gbm_regression" \
             "/interaction_simulation/k_50_sigma_5_x1_num"
    plot_fi_interaction(folder)

    # folder = "C:/Users/afeki/Desktop/code/CrossValidatedFeatureSelection/experiments/gbm_classification/adult/10Fold"
    # plot_fi('BARPLOT', folder, False)
