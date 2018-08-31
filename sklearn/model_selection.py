import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedShuffleSplit

from dvquys_python_utils.python_core.timeutils import timeit, update_progress

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

    @timeit
    def _eval(clf):
        scores = cross_val_score(estimator=clf, X=X, y=y, scoring=score, cv=cv)
        # print(scores.mean(), scores.std(), scores.mean() - scores.std() * 2, scores.mean() + scores.std() * 2)
        return scores
        
    for clf in estimators.values():
        clf_name = clf.__class__.__name__
        print("---\nSpotchecking for {}".format(clf_name))
        try:            
            scores, walltime = _eval(clf)
            print("{:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.0f}s".format(scores.mean(), scores.std(), scores.min(), scores.max(), walltime))
        except Exception as e:
            print(e.message)
            continue
        results[clf_name] = (
            scores.mean(), scores.std(), scores.mean() - scores.std() * 2, scores.mean() + scores.std() * 2, walltime
        )
        
    print("---")
    results_df = pd.DataFrame.from_dict(results, orient='index') \
        .rename(columns={0: 'mean',
                         1: 'std',
                         2: 'lower_bound',
                         3: 'upper_bound',
                         4: 'walltime(s)'}) \
        .sort_values(sort_by, ascending=False)
    return results_df

def split_train_test(X, y, test_size):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = 13)
    for i, j in sss.split(X, y):
        train_idx = i
        test_idx = j

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    return (X_train, y_train, X_test, y_test)