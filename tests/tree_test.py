import time

from numpy import random
from pandas import DataFrame, Series

from algorithms import CartRegressionTreeKFold, CartRegressionTree, TreeVisualizer
from tests.test_dataset_creator import create_x_y


# EXP = 'simulation'
# KFOLD = True
# MAX_DEPTH = 5
# tree = CartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else CartRegressionTree(max_depth=MAX_DEPTH)
# np.random.seed(3)
# X, y = create_x_y()
# print(np.mean(y[X['x1'] > 0]))
# print(np.mean(y[X['x1'] <= 0]))
# print(y.mean())
# start = time.time()
# tree.fit(X, y)
# end = time.time()
# print(end - start)
# tree_vis = TreeVisualizer()
# tree_vis.plot(tree)

def create_x_y(regression=True):
    df = DataFrame()
    n_rows = 10 ** 4
    n_numeric_cols = 10
    n_categorical_cols = 0
    n_categorical_values = 50
    for col in range(n_numeric_cols):
        df[col] = random.random(n_rows)
    for col in range(n_categorical_cols):
        df[col + n_numeric_cols] = random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
    y = random.random(n_rows) if regression else random.randint(2, size=n_rows)
    return df, Series(y)


if __name__ == "__main__":
    KFOLD = True
    MAX_DEPTH = 3
    tree = CartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else CartRegressionTree(max_depth=MAX_DEPTH)
    random.seed(10)
    X, y = create_x_y()
    start = time.time()
    tree.fit(X, y)
    end = time.time()
    print(end - start)
    tree_vis = TreeVisualizer()
    tree_vis.plot(tree)
