import xgboost as xgb
from numpy import mean, square, array, sqrt
from numpy.random import permutation
from pandas import Series, DataFrame
from sklearn.metrics import f1_score

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values, regression_error, \
    classification_error
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames, get_non_categorical_colnames


class XgboostGbmWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample, objective, compute_error):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.numeric_col_names = get_non_categorical_colnames(dtypes)
        self.variant = variant
        self.n_estimators = n_estimators
        self.param = {'max_depth': max_depth, 'eta': learning_rate, 'objective': objective, 'subsample': subsample}
        self.predictor = None
        self.x_train_cols = None
        self.compute_error = compute_error

    def fit(self, X, y):
        self.x_train_cols = X.columns
        dtrain = xgb.DMatrix(X, label=y)
        self.predictor = xgb.train(self.param, dtrain, self.n_estimators)

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
        fi = self.predictor.get_score(importance_type='gain')
        fi = Series(self.group_fi(fi))
        return normalize_series(fi)

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
        fi = Series(self.group_fi(results))
        return normalize_series(fi)

    def predict(self, X: DataFrame):
        return self.predictor.predict(xgb.DMatrix(X))

    def compute_fi_shap(self, X, y):
        # TODO: fix it
        fi = get_shap_values(self.predictor, X, self.x_train_cols).to_dict()
        fi = Series(self.group_fi(fi))
        return fi

    def n_leaves_per_tree(self):
        df = self.predictor.trees_to_dataframe()
        leaves_per_tree = df[df['Feature'] == 'Leaf']['Tree']
        n_leaves_per_tree = leaves_per_tree.value_counts()
        n_leaves_per_tree = n_leaves_per_tree[n_leaves_per_tree > 1]
        return n_leaves_per_tree

    def get_n_trees(self):
        return self.n_leaves_per_tree().size

    def get_n_leaves(self):
        return self.n_leaves_per_tree().sum()


class XgboostGbmRegressorWrapper(XgboostGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            objective='reg:squarederror',
            compute_error= regression_error)


class XgboostGbmClassifierWrapper(XgboostGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            objective='binary:logistic',
            compute_error= classification_error)
