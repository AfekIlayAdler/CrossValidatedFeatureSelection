from pandas import Series, DataFrame
from sklearn.datasets import load_breast_cancer

from experiments.config_object import Config
from experiments.default_config import RF_CLASSIFIERS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_num
from experiments.experiment_configurator import experiment_configurator

"""
Simple classification, only numerical features
"""


def get_x_y():
    data = load_breast_cancer()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    return X, y


if __name__ == '__main__':
    config = Config(
        kfold_flag=False,
        compute_permutation=False,
        save_results=True,
        one_hot=False,
        contains_num_features=True,
        seed=7,
        kfolds=30,
        predictors=RF_CLASSIFIERS,
        columns_to_remove=[],
        get_x_y=get_x_y,
        preprocessing_pipeline=get_preprocessing_pipeline_only_num)
    experiment_configurator(config)
