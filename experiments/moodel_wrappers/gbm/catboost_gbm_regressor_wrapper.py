from catboost import CatBoostRegressor, Pool
from numpy import mean, square, array, sqrt
from numpy.random import permutation
from pandas import Series

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames


class CatboostGbmRegressorWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, fast):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.variant = variant
        self.predictor = CatBoostRegressor(iterations=n_estimators,
                                           depth=max_depth,
                                           learning_rate=learning_rate,
                                           loss_function='RMSE', logging_level='Silent')

        self.x_train_cols = None

    def fit(self, X, y):
        self.x_train_cols = X.columns
        train_pool = Pool(X, y, cat_features=self.cat_col_indexes)
        self.predictor.fit(train_pool)

    def compute_fi_gain(self):
        # TODO: fix it
        fi = Series(self.predictor.feature_importances_, index=self.predictor.feature_names_)
        return normalize_series(fi)

    def get_pool(self, X):
        cat_features = self.cat_col_indexes if self.cat_col_indexes else None
        return Pool(X, cat_features=cat_features)

    def compute_fi_permutation(self, X, y):
        results = {}
        mse = mean(square(y - self.predictor.predict(self.get_pool(X))))
        for col in X.columns:
            permutated_x = X.copy()
            random_feature_mse = []
            for i in range(N_PERMUTATIONS):
                permutated_x[col] = permutation(permutated_x[col])
                temp_x = self.get_pool(permutated_x)
                random_feature_mse.append(mean(square(y - self.predictor.predict(temp_x))))
            results[col] = mean(array(random_feature_mse)) - mse
        fi = Series(results)
        return normalize_series(fi)

    def compute_fi_shap(self, X, y):
        # TODO: fix it
        pool = Pool(X, y, cat_features=self.cat_col_indexes)
        return get_shap_values(self.predictor, pool, self.x_train_cols)

    def compute_rmse(self, X, y):
        return sqrt(mean(square(y - self.predictor.predict(self.get_pool(X)))))

    def get_n_trees(self):
        return self.predictor.tree_count_

    def get_n_leaves(self):
        return self.predictor.tree_count_*(2**self.predictor._init_params['depth'])