import lightgbm as lgb
from numpy import mean, square, array
from numpy.random import permutation
from pandas import Series

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames, get_non_categorical_colnames


class LgbmGbmRegressorWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.numeric_col_names = get_non_categorical_colnames(dtypes)
        self.variant = variant
        self.n_estimators = n_estimators
        self.predictor = lgb.LGBMRegressor(
            boosting_type='gbdt',
            learning_rate=learning_rate,
            objective='regression',
            max_depth=max_depth,
            importance_type='gain')
        self.x_train_cols = None

    def fit(self, X, y):
        self.x_train_cols = X.columns
        self.predictor.fit(X, y, categorical_feature=self.cat_col_indexes)

    def group_fi(self, fi):
        if self.variant == 'one_hot':
            return_dict = {}
            for numeric_col in self.numeric_col_names:
                return_dict[numeric_col] = fi[numeric_col]
            for cat_col in self.cat_col_names:
                return_dict.setdefault(cat_col, 0)
                for k, v in fi.items():
                    if k.startswith(cat_col):
                        return_dict[cat_col] += fi[k]
            return return_dict
        return fi

    def compute_fi_gain(self):
        # TODO: fix it
        fi = dict(zip(self.x_train_cols ,self.predictor.feature_importances_))
        fi = Series(self.group_fi(fi))
        return normalize_series(fi)

    def compute_fi_permutation(self, X, y):
        results = {}
        mse = mean(square(y - self.predictor.predict(X)))
        # TODO: shuffling doesn't work here for some reason
        for col in X.columns:
            permutated_x = X.copy()
            random_feature_mse = []
            for i in range(N_PERMUTATIONS):
                permutated_x[col] = Series(permutation(permutated_x[col]), dtype = X[col].dtype)
                random_feature_mse.append(mean(square(y - self.predictor.predict(permutated_x))))
            results[col] = mean(array(random_feature_mse)) - mse
        fi = Series(self.group_fi(results))
        return normalize_series(fi)

    def compute_fi_shap(self, X, y):
        # TODO: fix it
        fi = get_shap_values(self.predictor, X, self.x_train_cols).to_dict()
        fi = Series(self.group_fi(fi))
        return fi
