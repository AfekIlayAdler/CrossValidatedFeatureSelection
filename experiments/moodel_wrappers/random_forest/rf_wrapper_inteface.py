class RfWrapperInterface:

    def fit(self, X, y):
        raise NotImplementedError

    def compute_fi_gain(self):
        raise NotImplementedError

    def compute_fi_permutation(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def compute_fi_shap(self, X, y):
        raise NotImplementedError

    def get_n_trees(self):
        raise NotImplementedError

    def get_n_leaves(self):
        raise NotImplementedError
