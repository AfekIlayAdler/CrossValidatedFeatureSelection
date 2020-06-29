from numpy import nan, mean, array, square
from numpy.random import permutation
from pandas import Series, DataFrame

from algorithms import FastCartRandomForestClassifier, FastCartKfoldRandomForestClassifier, \
    FastCartRandomForestRegressor, FastCartKfoldRandomForestRegressor, CartRandomForestClassifier, \
    CartKfoldRandomForestClassifier, CartRandomForestRegressor, CartKfoldRandomForestRegressor
from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, regression_error, classification_error


class OurRfWrapper:
    def __init__(self, model, compute_error):
        self.predictor = model
        self.x_train_cols = None
        self.compute_error = compute_error

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

    def get_n_trees(self):
        return self.predictor.n_estimators

    def get_n_leaves(self):
        return sum([tree.n_leaves for tree in self.predictor.trees])


class RegressionWrapper(OurRfWrapper):
    def __init__(self, model):
        super().__init__(model, compute_error=regression_error)

    def predict(self, X: DataFrame):
        return self.predictor.predict(X)


class ClassificationWrapper(OurRfWrapper):
    def __init__(self, model):
        super().__init__(model, compute_error=classification_error)

    def predict(self, X: DataFrame):
        return (self.predictor.predict(X) > 0.5) * 1

    def predict_proba(self, X: DataFrame):
        return self.predictor.predict(X)


"""Based on fast trees"""


class OurFastRfClassifierWrapper(ClassificationWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=FastCartRandomForestClassifier(n_estimators=n_estimators))


class OurFastKfoldRfClassifierWrapper(ClassificationWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=FastCartKfoldRandomForestClassifier(n_estimators=n_estimators))


class OurFastRfRegressorWrapper(RegressionWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=FastCartRandomForestRegressor(n_estimators=n_estimators))


class OurFastKfoldRfRegressorWrapper(RegressionWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=FastCartKfoldRandomForestRegressor(n_estimators=n_estimators))


"""Based on regular trees"""


class OurRfClassifierWrapper(ClassificationWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=CartRandomForestClassifier(n_estimators=n_estimators))


class OurKfoldRfClassifierWrapper(ClassificationWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=CartKfoldRandomForestClassifier(n_estimators=n_estimators))


class OurRfRegressorWrapper(RegressionWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=CartRandomForestRegressor(n_estimators=n_estimators))


class OurKfoldRfRegressorWrapper(RegressionWrapper):
    def __init__(self, variant, dtypes, n_estimators):
        super().__init__(model=CartKfoldRandomForestRegressor(n_estimators=n_estimators))
