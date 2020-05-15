from typing import Dict

from numpy import mean, square, sum, array, unique
from pandas import DataFrame, Series

from algorithms.Tree.config import PANDAS_CATEGORICAL_COLS


def get_cols_dtypes(df: DataFrame) -> Dict:
    return {i: v.name for i, v in df.dtypes.to_dict().items()}


def get_cat_cols(dtypes: Series):
    return dtypes[dtypes == 'category'].index.tolist()


def get_num_cols(dtypes: Series):
    return dtypes[dtypes != 'category'].index.tolist()


def get_cat_num_cols(dtypes):
    return get_cat_cols(dtypes), get_num_cols(dtypes)


def get_max_value_per_cat(X):
    # todo : make sure all cat values are in 0 - n-1 in preprocessing can remove the max and X[:, col].max()
    return array([max(len(unique(X[:, col])), X[:, col].max()) for col in range(X.shape[1])])


def get_col_type(col_type):
    if col_type in PANDAS_CATEGORICAL_COLS:
        return 'categorical'
    return 'numeric'


def regression_impurity(y: Series):
    return sum(square(y - mean(y)))


def classification_impurity(y: Series):
    p = sum(y) / y.size
    n = y.size
    return n * p * (1 - p)


impurity_dict = {'regression': regression_impurity, 'classification': classification_impurity}
