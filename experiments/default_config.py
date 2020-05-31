from pathlib import Path
from experiments.moodel_wrappers.gbm import *

from experiments.moodel_wrappers.random_forest import OurRfWrapperClassifier, OurRfWrapperRegressor, \
    SklearnRfRegressorWrapper, SklearnRfClassifierWrapper

# gbm
MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
SUBSAMPLE = 0.5

# io
# MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/")

# data
CATEGORY_COLUMN_NAME = 'category'
VAL_RATIO = 0.15
Y_COL_NAME = 'y'
N_ROWS = 10 ** 3

GBM_REGRESSORS = {
    'lgbm': LgbmGbmRegressorWrapper,
    'xgboost': XgboostGbmRegressorWrapper,
    'catboost': CatboostGbmRegressorWrapper,
    'sklearn': SklearnGbmRegressorWrapper,
    'ours_vanilla': OurFastGbmRegressorWrapper,
    'ours_kfold': OurFastKfoldGbmRegressorWrapper}

GBM_CLASSIFIERS = {
    'lgbm': LgbmGbmClassifierWrapper,
    'xgboost': XgboostGbmClassifierWrapper,
    'catboost': CatboostGbmClassifierWrapper,
    'sklearn': SklearnGbmClassifierWrapper,
    'ours_vanilla': OurFastGbmClassifierWrapper,
    'ours_kfold': OurFastKfoldGbmClassifierWrapper}

RF_REGRESSORS = {
    'sklearn': SklearnRfRegressorWrapper,
    'ours': OurRfWrapperRegressor}

RF_CLASSIFIERS = {
    'sklearn': SklearnRfClassifierWrapper,
    'ours': OurRfWrapperClassifier}
