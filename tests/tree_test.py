import time

from numpy import random

from algorithms import CartRegressionTreeKFold, CartRegressionTree, TreeVisualizer, node_based_feature_importance
from algorithms.Tree import FastCartRegressionTree, FastCartRegressionTreeKFold
from algorithms.Tree.utils import get_cat_num_cols
from tests.get_xy import get_x_y_boston, get_x_y_bike_rentals
from tests.test_dataset_creator import create_x_y

if __name__ == "__main__":
    KFOLD = False
    FAST = True
    DATA = 'BIKE_RANTAL' # BOSTON
    MAX_DEPTH = 3
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
        X = X#[cat_cols]
    else:
        X, y = create_x_y()
    start = time.time()
    tree.fit(X, y)
    end = time.time()
    print(end - start)
    print(node_based_feature_importance(tree))
    tree_vis = TreeVisualizer()
    tree_vis.plot(tree)
