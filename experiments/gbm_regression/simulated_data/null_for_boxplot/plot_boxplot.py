from experiments.plot_utils import get_regular_paths, get_data

BOXPLOT = True
BARPLOT = False
metrics = ['ntrees', 'nleaves', 'error']
paths = get_regular_paths(one_hot=False)
for model, model_path in paths.items():
    fi, metrics = get_data(model, model_path)
    if BOXPLOT:
        plot_boxplot(fi)
    # if BARPLOT:
    #     plot_barplot(fi)
    # plot_metrics(metrics)

