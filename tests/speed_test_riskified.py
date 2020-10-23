import time

from numpy import random
from pandas import DataFrame, Series

from experiments.moodel_wrappers.gbm import CatboostGbmRegressorWrapper, LgbmGbmRegressorWrapper


def create_x_y(regression=True):
    df = DataFrame()
    n_rows = 10 ** 6
    n_numeric_cols = 300
    n_categorical_cols = 0
    n_categorical_values = 30
    for col in range(n_numeric_cols):
        df[col] = random.random(n_rows)
    for col in range(n_categorical_cols):
        df[col + n_numeric_cols] = random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
    y = random.random(n_rows) if regression else random.randint(2, size=n_rows)
    return df, Series(y)


X, y = create_x_y()
reg = LgbmGbmRegressorWrapper('', X.dtypes, max_depth=6, n_estimators=200,
                                  learning_rate=0.01, subsample=1)

start = time.time()
reg.fit(X,y)
end = time.time()
print(end - start)

