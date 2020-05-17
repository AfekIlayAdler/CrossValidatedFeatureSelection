from pathlib import Path

from numpy import random
from pandas import DataFrame, Series, to_datetime, read_csv
from sklearn.datasets import load_boston


def get_x_y_boston():
    data = load_boston()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    return X, y


def get_x_y_bike_rentals():
    def preprocess_df(df):
        df['datetime'] = to_datetime(df['datetime']).dt.hour
        df['datetime'] = df['datetime'].astype('category')
        df['holiday'] = df['holiday'].astype(int)
        df['season'] = df['season'].astype('category')
        df['workingday'] = df['workingday'].astype('category')
        df['weather'] = df['weather'].astype('category')
        X = df[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']]
        y = df['count']
        return X, y

    project_root = Path(__file__).parent.parent
    train = read_csv(project_root / 'datasets/bike_rental_regression/train.csv')
    X, y = preprocess_df(train)
    return X, y


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



# def create_x_y(regression=True):
#     df = pd.DataFrame()
#     n_rows = 10 ** 3
#     n_numeric_cols = 0
#     n_categorical_cols = 50
#     n_categorical_values = 50
#     for col in range(n_numeric_cols):
#         df[col] = np.random.random(n_rows)
#     for col in range(n_categorical_cols):
#         df[col + n_numeric_cols] = np.random.randint(n_categorical_values, size=n_rows)
#         df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
#     y = np.random.random(n_rows) if regression else np.random.randint(2, size=n_rows)
#     return df, pd.Series(y)

# def create_x_y(category_size=52):
#     A1 = 3
#     A2 = 2
#     SIGMA = 10
#     N_ROWS = 10 ** 3
#     CATEGORY_COLUMN_NAME = 'random_category'
#     VAL_RATIO = 0.15
#     X = pd.DataFrame()
#     X['x1'] = np.random.randn(N_ROWS)
#     X['x2'] = np.random.randn(N_ROWS)
#     sigma = SIGMA * np.random.randn(N_ROWS)
#     y = A1 * X['x1'] + A2 * X['x2'] + sigma
#     X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
#     X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
#     return X, y


# def create_x_y():
#     a = 0.1
#     a = float(a)
#     N_ROWS = 1000
#     category_size = 10
#     CATEGORY_COLUMN_NAME = 'category'
#     X = pd.DataFrame()
#     X[CATEGORY_COLUMN_NAME] = np.random.randint(0, category_size, N_ROWS)
#     X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
#     X['x1'] = np.random.randn(N_ROWS)
#     sigma = 0.1 * np.random.randn(N_ROWS)
#     left_group = [i for i in range(category_size // 2)]
#     y = a * (X['x1'] > 0) + (1 - a) * X[CATEGORY_COLUMN_NAME].isin(left_group) + sigma
#     return X, y


# def create_x_y(regression=True):
#     df = pd.DataFrame()
#     n_rows = 10 ** 4
#     n_numeric_cols = 0
#     n_categorical_cols = 10
#     n_categorical_values = 10
#     for col in range(n_numeric_cols):
#         df[col] = np.random.random(n_rows)
#     for col in range(n_categorical_cols):
#         df[col + n_numeric_cols] = np.random.randint(n_categorical_values, size=n_rows)
#         df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
#     y = np.random.random(n_rows) if regression else np.random.randint(2, size=n_rows)
#     return df, pd.Series(y)



# def create_x_y():
#     a = 0.1
#     a = float(a)
#     N_ROWS = 1000
#     category_size = 10
#     CATEGORY_COLUMN_NAME = 'category'
#     X = DataFrame()
#     X[CATEGORY_COLUMN_NAME] = random.randint(0, category_size, N_ROWS)
#     X[CATEGORY_COLUMN_NAME] = X[CATEGORY_COLUMN_NAME].astype('category')
#     X['x1'] = random.randn(N_ROWS)
#     sigma = 0.1 * random.randn(N_ROWS)
#     left_group = [i for i in range(category_size // 2)]
#     y = a * (X['x1'] > 0) * 1 + (1 - a) * X[CATEGORY_COLUMN_NAME].isin(left_group) + sigma
#     return X, y