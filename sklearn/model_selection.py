import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

CLASSIFIERS = {
    'lr': LogisticRegression(),
    'mlp': MLPClassifier(),
    'naive': BernoulliNB(),
    'SVC': svm.SVC(),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'gb': GradientBoostingClassifier(),
    'xgb': XGBClassifier(),
    'dummy': DummyClassifier()
}



REGRESSORS = {
    'lr': LinearRegression(),
    'lasso': Lasso(),
    'ridge': Ridge(),
    'mlp': MLPRegressor(),
    'SVC': svm.SVR(),
    'knn': KNeighborsRegressor(),
    'rf': RandomForestRegressor(),
    'gb': GradientBoostingRegressor(),
    'xgb': XGBRegressor(),
    'dummy': DummyRegressor()
}


def spotcheck(estimators=CLASSIFIERS, X=None, y=None, score='roc_auc', cv=3, sort_by='mean'):
    print("Evaluation metrics: " + str(score))
    results = {}
    for clf in estimators.values():
        clf_name = clf.__class__.__name__
        print("---\nSpotchecking for {}".format(clf_name))
        try:
            scores = cross_val_score(estimator=clf, X=X, y=y, scoring=score, cv=cv)
        except Exception as e:
            print(e.message)
            continue
        print(scores.mean(), scores.std(), scores.min(), scores.max())
        # print(scores.mean(), scores.std(), scores.mean() - scores.std() * 2, scores.mean() + scores.std() * 2)
        results[clf_name] = (
        scores.mean(), scores.std(), scores.mean() - scores.std() * 2, scores.mean() + scores.std() * 2)
    print("---")
    results_df = pd.DataFrame.from_dict(results, orient='index') \
        .rename(columns={0: 'mean',
                         1: 'std',
                         2: 'lower_bound',
                         3: 'upper_bound'}) \
        .sort_values(sort_by, ascending=False)
    return results_df
