import jupyter as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

CLASSIFIERS = {
    'lr': LogisticRegression(),
    'mlp': MLPClassifier(),
    'naive': BernoulliNB(),
    'SVC': svm.SVC(),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'linearSVC': svm.LinearSVC(),
    'gb': GradientBoostingClassifier(),
    'xgb': XGBClassifier()
}


def spotcheck(classifiers=CLASSIFIERS, X=None, y=None, score='roc_auc', cv=3, sort_by='mean'):
    print("Evaluation metrics: " + str(score))
    results = {}
    for clf in classifiers.values():
        clf_name = clf.__class__.__name__
        print("---\nSpotchecking for {}".format(clf_name))
        try:
            scores = cross_val_score(estimator=clf, X=X, y=y, scoring=score, cv=cv)
        except Exception as e:
            print(e.message)
            continue
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
