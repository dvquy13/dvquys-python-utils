from sklearn.base import BaseEstimator, TransformerMixin
import jupyter as pd


AGE_GROUP_THRESHOLDS = [0,13,18,24,35,45,55,65,100]


class AgeBucketizer(BaseEstimator, TransformerMixin):
    def __init__(self, list_of_splits = AGE_GROUP_THRESHOLDS):
        self.splits = list_of_splits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.cut(X, self.splits, labels=False, include_lowest=True)
