from numpy import mean, square, array, sqrt, nan
from numpy.random import permutation
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.random_forest.rf_wrapper_inteface import RfWrapperInterface
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values, classification_error
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames, get_non_categorical_colnames


class SklearnRfWrapper(RfWrapperInterface):
    def __init__(self, variant, dtypes, model, compute_error):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.numeric_col_names = get_non_categorical_colnames(dtypes)
        self.variant = variant
        self.predictor = model
        self.x_train_cols = None
        self.compute_error = compute_error

    def fit(self, X, y):
        self.x_train_cols = X.columns
        self.predictor.fit(X, y)

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

    def predict(self, X: DataFrame):
        return self.predictor.predict(X)

    def compute_fi_gain(self):
        fi = dict(zip(self.x_train_cols, self.predictor.feature_importances_))
        fi = Series(self.group_fi(fi))
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
        # fi = get_shap_values(self.predictor, X, self.x_train_cols).to_dict()
        # fi = Series(self.group_fi(fi))
        return Series({col: nan for col in self.x_train_cols})

    def n_leaves_per_tree(self):
        n_leaves_per_tree = Series({i: tree.tree_.n_leaves for i, tree in enumerate(self.predictor.estimators_)})
        n_leaves_per_tree = n_leaves_per_tree[n_leaves_per_tree > 1]
        return n_leaves_per_tree

    def get_n_trees(self):
        return self.n_leaves_per_tree().size

    def get_n_leaves(self):
        return self.n_leaves_per_tree().sum()


class SklearnRfRegressorWrapper(SklearnRfWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        self.variant = variant
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            model=RandomForestRegressor(n_estimators=n_estimators),
            compute_error=classification_error)


class SklearnRfClassifierWrapper(SklearnRfWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        self.variant = variant
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            model=RandomForestClassifier(n_estimators=n_estimators),
            compute_error=classification_error)

    def predict_proba(self, X: DataFrame):
        return self.predictor.predict_proba(X)