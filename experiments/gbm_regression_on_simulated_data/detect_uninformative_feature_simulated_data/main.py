import multiprocessing

import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

from experiments.default_config import RESULTS_DIR, VAL_RATIO, MODELS_DIR, MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, \
    GBM_REGRSSORS, CATEGORY_COLUMN_NAME
from experiments.gbm_regression_on_simulated_data.detect_uninformative_feature_simulated_data.config import create_x_y, all_experiments, N_PROCESS, \
    n_total_experiments, MODELS, DEBUG
from experiments.utils import make_dirs, transform_categorical_features


def worker(model_name, variant, exp_number, category_size):
    exp_name = F"{model_name}_{variant}_exp_{exp_number}_category_size_{category_size}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
        return
    np.random.seed(exp_number)
    X, y = create_x_y(category_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    results = {'model': F"{model_name}_{variant}", 'categories': category_size, 'exp': exp_number}
    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, variant)
    model = GBM_REGRSSORS[model_name](variant, X.dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                      learning_rate=LEARNING_RATE)
    model.fit(X_train, y_train)
    results.update({
        'gain': model.compute_fi_gain()[CATEGORY_COLUMN_NAME],
        'permutation_train': model.compute_fi_permutation(X_train, y_train)[CATEGORY_COLUMN_NAME],
        'permutation_test': model.compute_fi_permutation(X_test, y_test)[CATEGORY_COLUMN_NAME],
        'shap_train': model.compute_fi_shap(X_train, y_train)[CATEGORY_COLUMN_NAME],
        'shap_test': model.compute_fi_shap(X_test, y_test)[CATEGORY_COLUMN_NAME]})
    DataFrame(Series(results)).T.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    print(f"n experimets for each model: {n_total_experiments}")
    for model_name, model_variants in MODELS.items():
        print(f'Working on experiment : {model_name}')
        args = []
        make_dirs([RESULTS_DIR / model_name])
        for exp_number, category_size in all_experiments():
            for variant in model_variants:
                if DEBUG:
                    worker(model_name, variant, exp_number, category_size)
                else:
                    args.append((model_name, variant, exp_number, category_size))
        if not DEBUG:
            with multiprocessing.Pool(N_PROCESS) as process_pool:
                process_pool.starmap(worker, args)
                # with concurrent.futures.ThreadPoolExecutor(4) as executor:
                #     results = list(tqdm(executor.map(lambda x: worker(*x), args), total=len(args)))
