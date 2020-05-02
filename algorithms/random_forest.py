import concurrent.futures

from numpy import random, sqrt, zeros

from .Tree import CartRegressionTree, CartClassificationTree, CartRegressionTreeKFold, CartClassificationTreeKFold, \
    MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, MIN_SAMPLES_SPLIT
from .config import N_FEATURES, BOOTSTRAP, N_ESTIMATORS, MAX_SAMPLES_FRACTION, RANDOM_STATE


def get_fitted_tree(self, x, y):
    tree = self.base_tree(min_samples_leaf=self.min_samples_leaf,
                          max_depth=self.max_depth,
                          min_impurity_decrease=self.min_impurity_decrease,
                          min_samples_split=self.min_samples_split)
    features = random.choice(self.n_cols, self.n_features, replace=False)
    replace = True if self.bootstrap else False
    rows = random.choice(self.n_rows, self.nrows_per_tree, replace=replace)
    temp_x, temp_y = x.iloc[rows, features], y[rows]
    tree.fit(temp_x, temp_y)
    return tree


class RandomForest:
    def __init__(self, base_tree,
                 n_features,
                 bootstrap,
                 n_estimators,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split,
                 max_samples_fraction,
                 random_state):
        self.base_tree = base_tree
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.max_samples_fraction = max_samples_fraction
        self.random_state = random_state
        self.n_rows, self.n_cols, self.nrows_per_tree = None, None, None
        self.trees = []

    def fit(self, x, y):
        self.n_rows, self.n_cols = x.shape
        self.nrows_per_tree = int(self.n_rows * self.max_samples_fraction)
        if self.random_state:
            random.seed(self.random_state)
        if self.n_features == "sqrt":
            self.n_features = int(sqrt(self.n_cols))

        with concurrent.futures.ProcessPoolExecutor() as e:
            results = [e.submit(get_fitted_tree, self, x, y) for _ in range(self.n_estimators)]
            self.trees = [f.result() for f in concurrent.futures.as_completed(results)]
        # self.trees = [get_fitted_tree(self, x, y) for _ in range(self.n_estimators)]

    def predict(self, x):
        prediction = zeros(x.shape[0])
        for tree in self.trees:
            prediction += tree.predict(x.to_dict('records'))
        return prediction / self.n_estimators


class CartRandomForestRegressor(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartRegressionTree,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class CartRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartClassificationTree,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class CartKfoldRandomForestRegressor(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class CartKfoldRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartClassificationTreeKFold,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)
