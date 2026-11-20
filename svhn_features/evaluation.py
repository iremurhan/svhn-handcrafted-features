"""Feature distinctiveness and classifier comparison.

Scaling and PCA are part of each scikit-learn Pipeline, so they are fitted
on the training split only and applied identically at test time.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def fisher_ratio(features, labels):
    """Between-class over within-class scatter (trace ratio) of standardised features.

    Higher means classes are more separated relative to their spread, which is a
    better proxy for distinctiveness than raw feature variance.
    """
    X = StandardScaler().fit_transform(features)
    mean = X.mean(axis=0)
    between = within = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        between += len(Xc) * np.sum((Xc.mean(axis=0) - mean) ** 2)
        within += np.sum((Xc - Xc.mean(axis=0)) ** 2)
    return between / max(within, 1e-12)


def make_models(seed=0, pca_components=100):
    def pipe(clf, pca=False):
        steps = [("scale", StandardScaler())]
        if pca:
            steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))
        return Pipeline(steps + [("clf", clf)])

    return {
        "knn": pipe(KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        "lda": pipe(LinearDiscriminantAnalysis()),
        "logreg": pipe(LogisticRegression(max_iter=2000)),
        "decision_tree": pipe(DecisionTreeClassifier(random_state=seed)),
        "decision_tree_pca": pipe(DecisionTreeClassifier(random_state=seed), pca=True),
        "random_forest": pipe(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
        "random_forest_pca": pipe(RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                         random_state=seed), pca=True),
    }


def fit_and_score(model, X_train, y_train, X_test, y_test):
    if "pca" in model.named_steps:
        n = min(model.named_steps["pca"].n_components, X_train.shape[1], X_train.shape[0])
        model.set_params(pca__n_components=n)
    model.fit(X_train, y_train)
    return {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
    }
