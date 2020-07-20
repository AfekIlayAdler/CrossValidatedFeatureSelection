from experiments.plot_utils import get_regular_paths, get_feature_importance, plot_fi, plot_metrics

if __name__ == '__main__':
    is_classification = True
    metrics = ['ntrees', 'nleaves', 'error']
    if is_classification:
        metrics += ['logloss']
    paths = get_regular_paths(one_hot=False)
    for model, model_path in paths.items():
        model_fi = get_feature_importance(model, model_path)
        plot_fi(model, model_fi)
    for metric in metrics:
        plot_metrics(paths, metric)