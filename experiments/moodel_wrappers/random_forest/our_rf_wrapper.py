from numpy import nan, mean, array, square
from numpy.random import permutation
from pandas import Series

from algorithms import CartKfoldRandomForestClassifier, CartRandomForestClassifier, CartKfoldRandomForestRegressor, \
    CartRandomForestRegressor
from experiments.moodel_wrappers.random_forest.rf_wrapper_inteface import RfWrapperInterface
from experiments.moodel_wrappers.wrapper_utils import normalize_series
from experiments.moodel_wrappers.models_config import N_PERMUTATIONS


class OurRfWrapper(RfWrapperInterface):
    def __init__(self):
        self.variant = None
        self.predictor = None
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


class OurRfWrapperClassifier(OurRfWrapper):

    def __init__(self, variant, n_estimators):
        self.variant = variant
        model = CartKfoldRandomForestClassifier if variant == 'kfold' else CartRandomForestClassifier
        self.predictor = model(n_estimators=n_estimators)


class OurRfWrapperRegressor(OurRfWrapper):

    def __init__(self, variant, n_estimators):
        self.variant = variant
        model = CartKfoldRandomForestRegressor if variant == 'kfold' else CartRandomForestRegressor
        self.predictor = model(n_estimators=n_estimators)


