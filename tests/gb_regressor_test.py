import time

import numpy as np
from sklearn.model_selection import train_test_split

from algorithms import CartGradientBoostingRegressorKfold, CartGradientBoostingRegressor, \
    FastCartGradientBoostingRegressorKfold, FastCartGradientBoostingRegressor
from algorithms.Tree import TreeVisualizer
from tests.get_xy import get_x_y_boston

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error

    np.seterr(all='raise')
    KFOLD = True
    FAST = True
    BOSTON = True
    BIKE_RENTALS = False
    MAX_DEPTH = 3
    np.random.seed(3)
    if FAST:
        model = FastCartGradientBoostingRegressorKfold if KFOLD else FastCartGradientBoostingRegressor
    else:
        model = CartGradientBoostingRegressorKfold if KFOLD else CartGradientBoostingRegressor
    reg = model(max_depth=3)

    start = time.time()
    X, y = get_x_y_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1, random_state=42)
    reg.fit(X_train, y_train)
    end = time.time()
    print(end - start)
    print(f"mse is {mean_squared_error(y_test, reg.predict(X_test))}")
    tree_vis = TreeVisualizer()
    tree_vis.plot(reg.trees[0])
    # tree_vis.plot(reg.trees[1].root)
    # tree_vis.plot(reg.trees[2].root)
    print(reg.n_trees)
    print(reg.compute_feature_importance())
