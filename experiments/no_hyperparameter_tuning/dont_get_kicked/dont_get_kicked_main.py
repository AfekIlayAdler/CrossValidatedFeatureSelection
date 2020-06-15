from pathlib import Path

from numpy import nan, unique, sum, isnan
from numpy.random import seed
from pandas import read_csv, Series, DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from algorithms import FastCartGradientBoostingClassifierKfold, FastCartGradientBoostingClassifier, \
    CartGradientBoostingClassifierKfold, CartGradientBoostingClassifier, TreeVisualizer
from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import VAL_RATIO
from experiments.moodel_wrappers.wrapper_utils import normalize_series
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'IsBadBuy'
    train = read_csv(project_root / 'datasets/dont_get_kicked_catboost/training.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    for col in train.select_dtypes(include=['O']).columns.tolist() + ['WheelTypeID', 'BYRNO', 'VNZIP1']:
        X[col] = X[col].astype('category')
    return X, y


def run_experiment(
        model_name: str, get_data: callable, compute_permutation: bool, \
        save_results: bool, model, exp_results_path):
    X, y = get_data()
    seed(7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    preprocessing_pipeline.fit(X_train)
    X_train = preprocessing_pipeline.transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    print("binning data")
    num_cols = get_num_cols(X_train.dtypes)
    bin_mapper = BinMapper(max_bins=256, random_state=42)
    X_train.loc[:, num_cols] = bin_mapper.fit_transform(X_train.loc[:, num_cols].values)
    X_test.loc[:, num_cols] = bin_mapper.transform(X_test.loc[:, num_cols].values)

    original_dtypes = X_train.dtypes
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    if compute_permutation:
        permutation_train = model.compute_fi_permutation(X_train, y_train).to_dict()
        permutation_test = model.compute_fi_permutation(X_test, y_test).to_dict()
    else:
        empty_dict = Series({col: nan for col in original_dtypes})
        permutation_train = empty_dict
        permutation_test = empty_dict

    is_classification = len(unique(y)) == 2
    if is_classification:
        df = DataFrame()
        df['p'] = model.predict(X_test)
        df['y'] = y_test
        df  = df[df.p.notna()]
        logloss = log_loss(df['y'], df['p'])
    else:
        logloss = nan

    fi = Series(model.compute_feature_importance(method='gain'))
    fi = normalize_series(fi).to_dict()
    results = dict(model=f"{model_name}", ntrees=len(model.trees),
                   leaves=[tree.n_leaves for tree in model.trees],
                   nleaves=sum([tree.n_leaves for tree in model.trees]),
                   logloss=logloss,
                   gain=fi)

    # if save_results:
    #     DataFrame(Series(results)).T.to_csv(exp_results_path)
    print(logloss)
    # for tree in model.trees:
    #     try:
    #         tree_vis = TreeVisualizer()
    #         tree_vis.plot(tree)
    #     except:
    #         return


if __name__ == '__main__':
    models = {
        'kfold_no_hyper': FastCartGradientBoostingClassifierKfold(max_depth=100,
                                                                  n_estimators=2,
                                                                  learning_rate=0.1),

        'kfold_height': FastCartGradientBoostingClassifierKfold(max_depth=3,
                                                                n_estimators=1000,
                                                                learning_rate=0.1),
        # 'vanilla_no_hyper': CartGradientBoostingClassifier(max_depth=100,
        #                                                    n_estimators=1000,
        #                                                    learning_rate=0.1),

        'vanilla_height': FastCartGradientBoostingClassifier(max_depth=3,
                                                             n_estimators=100,
                                                             learning_rate=0.1)}

    for exp_name, model in models.items():
        print(f"run exp {exp_name}")
        output_path = Path(f"{exp_name}.csv")
        preprocessing_pipeline = get_preprocessing_pipeline(0.5, ['PurchDate', 'RefId'])
        if output_path.exists():
            continue
        run_experiment(exp_name, get_x_y, False, True, model, output_path)
