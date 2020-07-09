from pathlib import Path

from numpy import arange
from pandas import read_csv

from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat
from experiments.experiment_configurator import experiment_configurator

"""
regression data set. original contains 81 features. I choose only the categorical ones.
1460 records
"""


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'SalePrice'
    train = read_csv(project_root / 'datasets/house_prices_kaggle/train.csv')
    y = train[y_col_name]
    X = train.drop(columns=[y_col_name])
    X = X[['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
           'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
           'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']]
    if WITH_INDEX:
        X['index'] = arange(X.shape[0])
        X['index'] = X['index'].astype('category')
    return X, y


if __name__ == '__main__':
    WITH_INDEX = False
    config = Config(
        kfold_flag=False,
        drop_one_feature_flag=True,
        compute_permutation=False,
        save_results=True,
        one_hot=not WITH_INDEX,
        contains_num_features=False,
        seed=7,
        predictors=GBM_REGRESSORS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline_only_cat)
    experiment_configurator(config)
