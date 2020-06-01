from typing import List

from numpy import number
from sklearn.pipeline import Pipeline, make_pipeline

from PandasSklearnPipeline.dataframe_transformers import NanColumnsRemover, ObjectsColumnaAsType, TypeSelector, \
    PandasImputer, PandasStandardScaler, ColumnRemover, CatToInt, ColAsCategory
from PandasSklearnPipeline.pandas_feature_union import PandasFeatureUnion


def get_preprocessing_pipeline(p: float, columns: List[str]):
    return Pipeline([
        ("RemoveColumns", ColumnRemover(columns)),
        ("NanRemover", NanColumnsRemover(p)),
        ("TransformObjectsToCatOrBool", ObjectsColumnaAsType()),
        ("ImputeAndTransform",
         PandasFeatureUnion(transformer_list=[
             ("numeric_features", make_pipeline(
                 TypeSelector(number),
                 PandasImputer(strategy="mean"),
                 PandasStandardScaler()
             )),
             ("categorical_features", make_pipeline(
                 TypeSelector("category"),
                 CatToInt(),
                 PandasImputer(strategy="most_frequent"),
                 ColAsCategory()

             )),
             ("boolean_features", make_pipeline(
                 TypeSelector("bool"),
             ))
         ]))
    ])


def get_preprocessing_pipeline_only_cat(p: float, columns: List[str]):
    return Pipeline([
        ("RemoveColumns", ColumnRemover(columns)),
        ("NanRemover", NanColumnsRemover(p)),
        ("TransformObjectsToCatOrBool", ObjectsColumnaAsType()),
        ("CatToInt", CatToInt()),
        ("Impute", PandasImputer(strategy="most_frequent")),
        ("ColAsInt", ColAsCategory())])


def get_preprocessing_pipeline_only_num(p: float, columns: List[str]):
    return Pipeline([
        ("RemoveColumns", ColumnRemover(columns)),
        ("NanRemover", NanColumnsRemover(p)),
        ("TransformObjectsToCatOrBool", ObjectsColumnaAsType()),
        ("Impute", PandasImputer(strategy="mean")),
        ("Scale", PandasStandardScaler())])
