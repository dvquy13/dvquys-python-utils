import pandas as pd
from sklearn.model_selection import cross_val_score


def rfe(estimator, X, y, step=1, acceptable_decrease=0.01, cv=None, scoring=None, negletible_importance_threshold=0.001):
    """
    XGBoost feature selection based on feature_importances_
    :param estimator:
    :param X:
    :param y:
    :param step:
    :param acceptable_decrease:
    :param cv:
    :param scoring:
    :param negletible_importance_threshold:
    :return: logs as a pandas DataFrame, features to remove
    """

    estimator.fit(X, y)
    feat_imp = pd.DataFrame(data=estimator.feature_importances_, index=X.columns, columns=['importance'])\
        .sort_values(by='importance', ascending=False)
    zero_imp = feat_imp[feat_imp['importance'] < negletible_importance_threshold].index
    X = X.drop(zero_imp, axis=1)
    estimator.fit(X, y)
    score_full_f = cross_val_score(estimator, X, y, cv=cv, scoring=scoring).mean()
    new_score = score_full_f
    loop_logs = {}
    i = 0
    while (score_full_f - new_score) < acceptable_decrease:
        i += 1
        new_feat_imp = pd.DataFrame(data=estimator.feature_importances_, index=X.columns, columns=['importance'])\
            .sort_values(by='importance', ascending=False)
        worst_f = new_feat_imp.iloc[-step:].index
        X = X.drop(worst_f, axis=1)
        new_score = cross_val_score(estimator, X, y, cv=cv, scoring=scoring).mean()
        loop_logs[i] = {'remove': worst_f[0], 'new_score': new_score}
        estimator.fit(X, y)
    res = pd.DataFrame(loop_logs.values())
    res.index = res.index + len(zero_imp) + 1
    rfe_remove_f = res.iloc[:-1]['remove'].tolist() + zero_imp.tolist()
    return res, rfe_remove_f
