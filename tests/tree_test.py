import time
from pathlib import Path

from numpy import random
from pandas import read_csv

from algorithms import CartRegressionTreeKFold, CartRegressionTree, TreeVisualizer, node_based_feature_importance
from algorithms.Tree import FastCartRegressionTree, FastCartRegressionTreeKFold
from algorithms.Tree.utils import get_cat_num_cols
from tests.get_xy import get_x_y_boston, get_x_y_bike_rentals, get_x_y_amazon, get_x_y_increasing_size
from tests.get_xy import create_x_y

if __name__ == "__main__":
    KFOLD = True
    FAST = True
    DATA = 'SIMULATED_INCREASING_ORDER' # 'AMAZON' , 'AMAZON', 'BIKE_RANTAL', 'BOSTON'
    MAX_DEPTH = 5
    if FAST:
        model = FastCartRegressionTreeKFold if KFOLD else FastCartRegressionTree
    else:
        model = CartRegressionTreeKFold if KFOLD else CartRegressionTree

    tree = model(max_depth=MAX_DEPTH)
    random.seed(10)

    if DATA == 'BOSTON':
        X, y = get_x_y_boston()
    elif DATA == 'BIKE_RANTAL':
        X, y = get_x_y_bike_rentals()
        cat_cols, num_cols = get_cat_num_cols(X.dtypes)
        X = X  # [cat_cols]
    elif DATA == 'AMAZON':
        X, y = get_x_y_amazon()
    elif DATA == 'SIMULATED_INCREASING_ORDER':
        X, y = get_x_y_increasing_size()
    else:
        X, y = create_x_y()

    start = time.time()
    tree.fit(X, y)
    end = time.time()
    print(end - start)
    print(node_based_feature_importance(tree))
    tree_vis = TreeVisualizer()
    tree_vis.plot(tree)
