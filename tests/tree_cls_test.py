from ml_algorithms.classifiers import NumericDecisionTreeClassifier
import numpy as np

def test_gini_pure():
    clf = NumericDecisionTreeClassifier()
    assert clf._gini(np.array([1, 1, 1])) == 0.0

def test_gini_equal_classes():
    clf = NumericDecisionTreeClassifier()
    assert np.isclose(clf._gini(np.array([0, 1])), 0.5)

def test_gini_imbalanced():
    clf = NumericDecisionTreeClassifier()
    assert np.isclose(clf._gini(np.array([0, 0, 1])), 0.4444444444)

def test_simple_fit_predict():
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf = NumericDecisionTreeClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    print(preds)
    assert (preds == y).all()

def test_max_depth_stops_growth():
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf = NumericDecisionTreeClassifier(max_depth=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(np.unique(preds)) == 1
