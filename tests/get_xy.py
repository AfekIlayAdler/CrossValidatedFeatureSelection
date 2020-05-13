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
