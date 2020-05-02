import multiprocessing

import numpy as np
from pandas import Series, DataFrame, read_csv
from sklearn.model_selection import train_test_split

from experiments.default_config import RESULTS_DIR, VAL_RATIO, MODELS_DIR, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, \
    CATEGORY_COLUMN_NAME, RF_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.rf_classification_house_grants.config_rf_classification_house_grants import DATA_PATH, \
    Y_COL_NAME, MODELS, N_EXPERIMENTS, DEBUG, N_PROCESS, PROPORTION_NAN_COL_REMOVE, COLUMNS_TO_REMOVE
from experiments.utils import make_dirs, transform_categorical_features


def get_x_y():
    df = read_csv(DATA_PATH)
    y = df[Y_COL_NAME]
    X = df.drop(columns=[Y_COL_NAME])
    preprocessing_pipeline = get_preprocessing_pipeline(PROPORTION_NAN_COL_REMOVE, COLUMNS_TO_REMOVE)
    X = preprocessing_pipeline.fit_transform(X)
    return X, y


def worker(model_name, variant, exp_number):
    exp_name = F"{model_name}_{variant}_exp_{exp_number}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
        return
    np.random.seed(exp_number)
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    results = {'model': F"{model_name}_{variant}", 'exp': exp_number}
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = RF_REGRESSORS[model_name](variant, X.dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                      learning_rate=LEARNING_RATE)
    model.fit(X_train, y_train)
    print("finished fittin the model")
    results.update({
        'gain': model.compute_fi_gain(),
        'permutation_train': model.compute_fi_permutation(X_train, y_train),
        'permutation_test': model.compute_fi_permutation(X_test, y_test),
        'shap_train': model.compute_fi_shap(X_train, y_train),
        'shap_test': model.compute_fi_shap(X_test, y_test)})
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    print(f"n experimets for each model: {N_EXPERIMENTS}")
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        args = []
        make_dirs([RESULTS_DIR / model_name])
        for exp_number in range(N_EXPERIMENTS):
            for variant in model_variants:
                if DEBUG:
                    worker(model_name, variant, exp_number)
                else:
                    args.append((model_name, variant, exp_number))
        if not DEBUG:
            with multiprocessing.Pool(N_PROCESS) as process_pool:
                process_pool.starmap(worker, args)
                # with concurrent.futures.ThreadPoolExecutor(4) as executor:
                #     results = list(tqdm(executor.map(lambda x: worker(*x), args), total=len(args)))
