from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from numpy import mean, array
from numpy.random import permutation
from pandas import Series, DataFrame

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values, regression_error, \
    classification_error
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames


class CatboostGbmWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample, model, compute_error):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.variant = variant
        if model == 'regression':
            # I have encounterd slightly different behaviour so added it to be sure
            if subsample != 1.0:
                self.predictor = CatBoostRegressor(iterations=n_estimators,
                                                   depth=max_depth,
                                                   learning_rate=learning_rate,
                                                   loss_function='RMSE', logging_level='Silent', subsample=subsample,
                                                   bootstrap_type='Bernoulli')
            else:
                self.predictor = CatBoostRegressor(iterations=n_estimators,
                                                   depth=max_depth,
                                                   learning_rate=learning_rate,
                                                   loss_function='RMSE', logging_level='Silent')
        else:
            if subsample != 1.0:
                self.predictor = CatBoostClassifier(iterations=n_estimators,
                                                    depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    logging_level='Silent', subsample=subsample,
                                                    bootstrap_type='Bernoulli')
            else:
                self.predictor = CatBoostClassifier(iterations=n_estimators,
                                                    depth=max_depth,
                                                    learning_rate=learning_rate, logging_level='Silent')

        self.x_train_cols = None
        self.compute_error = compute_error

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
        true_error = self.compute_error(y, self.predict(X))
        for col in X.columns:
            permutated_x = X.copy()
            random_feature_mse = []
            for i in range(N_PERMUTATIONS):
                permutated_x[col] = permutation(permutated_x[col])
                random_feature_mse.append(self.compute_error(y, self.predict(permutated_x)))
            results[col] = mean(array(random_feature_mse)) - true_error
        fi = Series(results)
        return normalize_series(fi)

    def predict(self, X: DataFrame):
        return self.predictor.predict(self.get_pool(X))

    def compute_fi_shap(self, X, y):
        # TODO: fix it
        pool = Pool(X, y, cat_features=self.cat_col_indexes)
        return get_shap_values(self.predictor, pool, self.x_train_cols)

    def get_n_trees(self):
        return self.predictor.tree_count_

    def get_n_leaves(self):
        return self.predictor.tree_count_ * (2 ** self.predictor.init_params['depth'])


class CatboostGbmRegressorWrapper(CatboostGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model='regression',
            compute_error=regression_error)


class CatboostGbmClassifierWrapper(CatboostGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model='classification',
            compute_error=classification_error)
