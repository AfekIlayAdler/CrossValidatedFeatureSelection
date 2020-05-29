import multiprocessing

from numpy import mean, square, array, nan, sqrt
from numpy.random import permutation
from pandas import Series
from sklearn.metrics import f1_score

from algorithms import CartGradientBoostingRegressorKfold, CartGradientBoostingRegressor, \
    FastCartGradientBoostingRegressorKfold, FastCartGradientBoostingRegressor, CartGradientBoostingClassifier, \
    CartGradientBoostingClassifierKfold, FastCartGradientBoostingClassifier, FastCartGradientBoostingClassifierKfold
from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series


def worker(X, y, col, predict):
    permutated_x = X.copy()
    permutated_x[col] = permutation(permutated_x[col])
    return mean(square(y - predict(permutated_x)))


class OurGbmWrapper:
    def __init__(self, max_depth, n_estimators,
                 learning_rate, subsample, model):
        self.predictor = model(max_depth=max_depth, n_estimators=n_estimators,
                               learning_rate=learning_rate, subsample=subsample, min_samples_leaf=5)
        self.x_train_cols = None

    def fit(self, X, y):
        self.x_train_cols = X.columns
        self.predictor.fit(X, y)

    def compute_fi_gain(self):
        fi = Series(self.predictor.compute_feature_importance(method='gain'))
        return normalize_series(fi)

    def compute_fi_permutation(self, X, y):
        results = {}
        mse = mean(square(y - self.predictor.predict(X)))
        for col in X.columns:
            args = [(X, y, col, self.predictor.predict) for _ in range(N_PERMUTATIONS)]

            with multiprocessing.Pool() as process_pool:
                prm_results = process_pool.starmap(worker, args)

            results[col] = mean(array(prm_results)) - mse
            print(col)
        fi = Series(results)
        return normalize_series(fi)

    def compute_fi_shap(self, X, y):
        return Series({col: nan for col in self.x_train_cols})

    def compute_rmse(self, X, y):
        return sqrt(mean(square(y - self.predictor.predict(X))))

    def compute_f1(self, X, y):
        return f1_score(y, (self.predictor.predict(X) > 0.5) * 1)

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
            model=CartGradientBoostingRegressor)


class OurKfoldGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingRegressorKfold)


class OurFastGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingRegressor)


class OurFastKfoldGbmRegressorWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingRegressorKfold)


class OurGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingClassifier)


class OurKfoldGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=CartGradientBoostingClassifierKfold)


class OurFastGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingClassifier)


class OurFastKfoldGbmClassifierWrapper(OurGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=FastCartGradientBoostingClassifierKfold)
