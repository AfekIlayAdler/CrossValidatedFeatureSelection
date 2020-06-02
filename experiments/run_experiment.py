from time import time

from numpy import nan, unique
from numpy.random import seed
from pandas import Series, DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from algorithms import LEARNING_RATE
from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import VAL_RATIO, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, RESULTS_DIR
from experiments.utils import transform_categorical_features, make_dirs


def run_experiments(config):
    if config.one_hot:
        variants = ['mean_imputing', 'one_hot']
    else:
        variants = ['mean_imputing']

    models = dict(lgbm=['vanilla'], xgboost=variants, catboost=['vanilla'], sklearn=variants,
                  ours_vanilla=['_'], ours_kfold=['_'])

    start = time()
    make_dirs([RESULTS_DIR])
    for model_name, model_variants in models.items():
        print(f'Working on experiment : {model_name}')
        exp_dir = RESULTS_DIR / model_name
        make_dirs([exp_dir])
        for variant in model_variants:
            exp_name = F"{model_name}_{variant}.csv"
            exp_results_path = exp_dir / exp_name
            if exp_results_path.exists():
                continue

            seed(config.seed)
            run_experiment(
                model_name=model_name,
                variant=variant,
                get_data=config.get_x_y,
                compute_permutation=config.compute_permutation,
                save_results=config.save_results,
                contains_num_features=config.contains_num_features,
                preprocessing_pipeline=config.preprocessing_pipeline(0.5, config.columns_to_remove),
                models_dict=config.predictors,
                exp_results_path=exp_results_path)

    end = time()
    print(f"run took {end - start} seconds")


def run_experiment(
        model_name: str, variant: str, get_data: callable, compute_permutation: bool, \
        save_results: bool, contains_num_features: bool, preprocessing_pipeline, models_dict,
        exp_results_path):
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO)
    preprocessing_pipeline.fit(X_train)
    X_train = preprocessing_pipeline.transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)
    original_dtypes = X_train.dtypes
    if contains_num_features and model_name.startswith('ours'):
        print("binning data")
        num_cols = get_num_cols(original_dtypes)
        bin_mapper = BinMapper(max_bins=256, random_state=42)
        X_train.loc[:, num_cols] = bin_mapper.fit_transform(X_train.loc[:, num_cols].values)
        X_test.loc[:, num_cols] = bin_mapper.transform(X_test.loc[:, num_cols].values)

    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = models_dict[model_name](variant, original_dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                    learning_rate=LEARNING_RATE, subsample=1.)
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
        probabiliteis = model.predict_proba(X_test)
        logloss = log_loss(y_test, probabiliteis)
    else:
        logloss = nan
    results = dict(model=f"{model_name}_{variant}", ntrees=model.get_n_trees(), nleaves=model.get_n_leaves(),
                   error=model.compute_error(y_test, test_prediction), logloss=logloss,
                   gain=model.compute_fi_gain().to_dict(),
                   permutation_train=permutation_train, permutation_test=permutation_test,
                   shap_train=model.compute_fi_shap(X_train, y_train).to_dict(),
                   shap_test=model.compute_fi_shap(X_test, y_test).to_dict(),
                   test_labels=y_test.values.tolist(), predictions=test_prediction)

    if save_results:
        DataFrame(Series(results)).T.to_csv(exp_results_path)
