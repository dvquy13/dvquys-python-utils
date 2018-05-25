from sklearn.base import BaseEstimator, TransformerMixin
import jupyter as pd


class PandasFactorizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        selection = X.apply(lambda p: pd.factorize(p)[0] + 1)
        return selection.values


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names, factorize=False):
        self.attribute_names = attribute_names
        self.factorize = factorize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        selection = X[self.attribute_names]
        if self.factorize:
            selection = selection.apply(lambda p: pd.factorize(p)[0] + 1)
        return selection.values


