from collections import OrderedDict

import lightgbm as lgb
from numpy import mean, square, array, sqrt
from numpy.random import permutation
from pandas import Series, DataFrame
from sklearn.metrics import f1_score

from experiments.moodel_wrappers.models_config import N_PERMUTATIONS
from experiments.moodel_wrappers.wrapper_utils import normalize_series, get_shap_values
from experiments.utils import get_categorical_col_indexes, get_categorical_colnames, get_non_categorical_colnames

"""
not released yet.
taken from https://github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py
"""


def trees_to_dataframe(self):
    """Parse the fitted model and return in an easy-to-read pandas DataFrame.
    Returns
    -------
    result : pandas DataFrame
        Returns a pandas DataFrame of the parsed model.
    """

    def _is_split_node(tree):
        return 'split_index' in tree.keys()

    def create_node_record(tree, node_depth=1, tree_index=None,
                           feature_names=None, parent_node=None):

        def _get_node_index(tree, tree_index):
            tree_num = str(tree_index) + '-' if tree_index is not None else ''
            is_split = _is_split_node(tree)
            node_type = 'S' if is_split else 'L'
            # if a single node tree it won't have `leaf_index` so return 0
            node_num = str(tree.get('split_index' if is_split else 'leaf_index', 0))
            return tree_num + node_type + node_num

        def _get_split_feature(tree, feature_names):
            if _is_split_node(tree):
                if feature_names is not None:
                    feature_name = feature_names[tree['split_feature']]
                else:
                    feature_name = tree['split_feature']
            else:
                feature_name = None
            return feature_name

        def _is_single_node_tree(tree):
            return set(tree.keys()) == {'leaf_value'}

        # Create the node record, and populate universal data members
        node = OrderedDict()
        node['tree_index'] = tree_index
        node['node_depth'] = node_depth
        node['node_index'] = _get_node_index(tree, tree_index)
        node['left_child'] = None
        node['right_child'] = None
        node['parent_index'] = parent_node
        node['split_feature'] = _get_split_feature(tree, feature_names)
        node['split_gain'] = None
        node['threshold'] = None
        node['decision_type'] = None
        node['missing_direction'] = None
        node['missing_type'] = None
        node['value'] = None
        node['weight'] = None
        node['count'] = None

        # Update values to reflect node type (leaf or split)
        if _is_split_node(tree):
            node['left_child'] = _get_node_index(tree['left_child'], tree_index)
            node['right_child'] = _get_node_index(tree['right_child'], tree_index)
            node['split_gain'] = tree['split_gain']
            node['threshold'] = tree['threshold']
            node['decision_type'] = tree['decision_type']
            node['missing_direction'] = 'left' if tree['default_left'] else 'right'
            node['missing_type'] = tree['missing_type']
            node['value'] = tree['internal_value']
            node['weight'] = tree['internal_weight']
            node['count'] = tree['internal_count']
        else:
            node['value'] = tree['leaf_value']
            if not _is_single_node_tree(tree):
                node['weight'] = tree['leaf_weight']
                node['count'] = tree['leaf_count']

        return node

    def tree_dict_to_node_list(tree, node_depth=1, tree_index=None,
                               feature_names=None, parent_node=None):

        node = create_node_record(tree,
                                  node_depth=node_depth,
                                  tree_index=tree_index,
                                  feature_names=feature_names,
                                  parent_node=parent_node)

        res = [node]

        if _is_split_node(tree):
            # traverse the next level of the tree
            children = ['left_child', 'right_child']
            for child in children:
                subtree_list = tree_dict_to_node_list(
                    tree[child],
                    node_depth=node_depth + 1,
                    tree_index=tree_index,
                    feature_names=feature_names,
                    parent_node=node['node_index'])
                # In tree format, "subtree_list" is a list of node records (dicts),
                # and we add node to the list.
                res.extend(subtree_list)
        return res

    model_dict = self.dump_model()
    feature_names = model_dict['feature_names']
    model_list = []
    for tree in model_dict['tree_info']:
        model_list.extend(tree_dict_to_node_list(tree['tree_structure'],
                                                 tree_index=tree['tree_index'],
                                                 feature_names=feature_names))

    return DataFrame(model_list, columns=model_list[0].keys())


class LgbmGbmWrapper:
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample, model):
        self.cat_col_indexes = get_categorical_col_indexes(dtypes)
        self.cat_col_names = get_categorical_colnames(dtypes)
        self.numeric_col_names = get_non_categorical_colnames(dtypes)
        self.variant = variant
        self.n_estimators = n_estimators
        self.predictor = model(
            learning_rate=learning_rate,
            max_depth=max_depth,
            importance_type='gain',
            bagging_freq=1,
            bagging_fraction=subsample)
        self.x_train_cols = None

    def fit(self, X, y):
        self.x_train_cols = X.columns
        self.predictor.fit(X, y, categorical_feature=self.cat_col_indexes)

    def group_fi(self, fi):
        if self.variant == 'one_hot':
            return_dict = {}
            for numeric_col in self.numeric_col_names:
                return_dict[numeric_col] = fi[numeric_col]
            for cat_col in self.cat_col_names:
                return_dict.setdefault(cat_col, 0)
                for k, v in fi.items():
                    if k.startswith(cat_col):
                        return_dict[cat_col] += fi[k]
            return return_dict
        return fi

    def compute_fi_gain(self):
        # TODO: fix it
        fi = dict(zip(self.x_train_cols, self.predictor.feature_importances_))
        fi = Series(self.group_fi(fi))
        return normalize_series(fi)

    def compute_fi_permutation(self, X, y):
        results = {}
        mse = mean(square(y - self.predictor.predict(X)))
        # TODO: shuffling doesn't work here for some reason
        for col in X.columns:
            permutated_x = X.copy()
            random_feature_mse = []
            for i in range(N_PERMUTATIONS):
                permutated_x[col] = Series(permutation(permutated_x[col]), dtype=X[col].dtype)
                random_feature_mse.append(mean(square(y - self.predictor.predict(permutated_x))))
            results[col] = mean(array(random_feature_mse)) - mse
        fi = Series(self.group_fi(results))
        return normalize_series(fi)

    def compute_fi_shap(self, X, y):
        # TODO: fix it
        fi = get_shap_values(self.predictor, X, self.x_train_cols).to_dict()
        fi = Series(self.group_fi(fi))
        return fi

    def compute_rmse(self, X, y):
        return sqrt(mean(square(y - self.predictor.predict(X))))

    def compute_f1(self, X, y):
        return f1_score(y, (self.predictor.predict(X) > 0.5) * 1)

    def n_leaves_per_tree(self):
        df = trees_to_dataframe(self.predictor.booster_)
        leaves_per_tree = df[df['split_feature'].isna()]['tree_index']
        n_leaves_per_tree = leaves_per_tree.value_counts()
        n_leaves_per_tree = n_leaves_per_tree[n_leaves_per_tree > 1]
        return n_leaves_per_tree

    def get_n_trees(self):
        return self.n_leaves_per_tree().size

    def get_n_leaves(self):
        return self.n_leaves_per_tree().sum()


class LgbmGbmRegressorWrapper(LgbmGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=lgb.LGBMRegressor)


class LgbmGbmClassifierWrapper(LgbmGbmWrapper):
    def __init__(self, variant, dtypes, max_depth, n_estimators,
                 learning_rate, subsample):
        super().__init__(
            variant=variant,
            dtypes=dtypes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            model=lgb.LGBMClassifier)
