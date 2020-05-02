
class RfWrapperInterface:

    def fit(self, X, y):
        raise NotImplementedError

    def compute_fi_gain(self):
        raise NotImplementedError

    def compute_fi_permutation(self, X, y):
        raise NotImplementedError

    def compute_fi_shap(self, X, y):
        raise NotImplementedError