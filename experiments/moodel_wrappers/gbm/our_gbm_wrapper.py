import multiprocessing
from time import time

from numpy import mean, array, nan
from numpy.random import permutation
from pandas import Series, DataFrame

from algorithms import CartGradientBoostingRegressorKfold, CartGradientBoostingRegressor, \
    FastCartGradientBoostingRegressorKfold, FastCartGradientBoostingRegressor, CartGradientBoostingClassifier, \
    CartGradientBoostingClassifierKfold, FastCartGradientBoostingClassifier, FastCartGradientBoostingClassifierKfold
from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, regression_error, classification_error


def worker(X, y, col, predict, compute_error):
    permutated_x = X.copy()
    permutated_x[col] = permutation(permutated_x[col])
    return compute_error(y,predict(permutated_x))


class OurGbmWrapper:
    def __init__(self, max_depth, n_estimators,
                 learning_rate, subsample, model, compute_error):
        self.predictor = model(max_depth=max_depth, n_estimators=n_estimators,
                               learning_rate=learning_rate, subsample=subsample, min_samples_leaf=5)
        self.x_train_cols = None
        self.compute_error = compute_error

    def fit(self, X, y):
        self.x_train_cols = X.columns
        self.predictor.fit(X, y)

    def compute_fi_gain(self):
        fi = Series(self.predictor.compute_feature_importance(method='gain'))
        return normalize_series(fi)

    def compute_fi_permutation(self, X, y):
        results = {col: 0 for col in self.x_train_cols}
        true_error = self.compute_error(y, self.predict(X))
        # get only features that got positive fi_gain
        gain_fi = self.compute_fi_gain()
        positive_fi_gain = gain_fi[gain_fi > 0].index.tolist()
        for col in positive_fi_gain:
            start = time()
            args = [(X, y, col, self.predict, self.compute_error) for _ in range(N_PERMUTATIONS)]

            with multiprocessing.Pool(4) as process_pool:
                prm_results = process_pool.starmap(worker, args)

            results[col] = mean(array(prm_results)) - true_error
            end = time()
            print(f"{col} run took {end - start}")
        fi = Series(results)
        return normalize_series(fi)

    def predict(self, X: DataFrame):
        return self.predictor.predict(X)

    def compute_fi_shap(self, X, y):
        return Series({col: nan for col in self.x_train_cols})

    def get_n_trees(self):
        return self.predictor.n_trees

    def get_n_leaves(self):
        return sum([tree.n_leaves for tree in self.predictor.trees])


class OurGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingRegressor,
            compute_error=regression_error
        )


class OurKfoldGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingRegressorKfold,
            compute_error=regression_error)


class OurFastGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingRegressor,
            compute_error=regression_error)


class OurFastKfoldGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingRegressorKfold,
            compute_error=regression_error)


class OurGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingClassifier,
            compute_error=classification_error)


class OurKfoldGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingClassifierKfold,
            compute_error=classification_error)


class OurFastGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingClassifier,
            compute_error=classification_error)


class OurFastKfoldGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingClassifierKfold,
            compute_error=classification_error)
