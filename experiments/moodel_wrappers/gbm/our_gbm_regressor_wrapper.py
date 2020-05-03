from numpy import mean, square, array, nan, sqrt
from numpy.random import permutation
from pandas import Series

from algorithms import CartGradientBoostingRegressorKfold, CartGradientBoostingRegressor
from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series


class OurGbmRegressorWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate):

        self.variant = variant
        model = CartGradientBoostingRegressorKfold if variant == 'Kfold' else CartGradientBoostingRegressor
        self.predictor = model(max_depth=max_depth, n_estimators=n_estimators,
                               learning_rate=learning_rate, min_samples_leaf=5)
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
            permutated_x = X.copy()
            random_feature_mse = []
            for i in range(N_PERMUTATIONS):
                permutated_x[col] = permutation(permutated_x[col])
                random_feature_mse.append(mean(square(y - self.predictor.predict(permutated_x))))
            results[col] = mean(array(random_feature_mse)) - mse
        fi = Series(results)
        return normalize_series(fi)

    def compute_fi_shap(self, X, y):
        return Series({col: nan for col in self.x_train_cols})

    def compute_rmse(self, X, y):
        return sqrt(mean(square(y - self.predictor.predict(X))))

    def get_n_trees(self):
        return self.predictor.n_trees

    def get_n_leaves(self):
        return sum([tree.n_leaves for tree in self.predictor.trees])
