from ml_algorithms.classifiers import NumericRandomForestClassifier
import numpy as np

def test_number_of_estimators():
    clf = NumericRandomForestClassifier(n_estimators=10)
    clf.fit(np.random.rand(100, 5), np.random.randint(0, 2, size=100))
    assert len(clf.estimators) == 10


def test_bootstrap_sampling():
    X = np.arange(100).reshape(50, 2)
    y = np.zeros(50).astype(int)
    clf = NumericRandomForestClassifier(n_estimators=1)
    clf.fit(X, y)

    unique_counts = np.unique(clf.estimators[0].X, axis=0).shape[0]
    assert unique_counts < X.shape[0]


def test_bootstrap_sampling():
    X = np.arange(100).reshape(50, 2)
    y = np.zeros(50).astype(int)
    clf = NumericRandomForestClassifier(n_estimators=1)
    clf.fit(X, y)

    unique_counts = np.unique(clf.estimators[0]._X, axis=0).shape[0]
    assert unique_counts < X.shape[0]


def test_consistent_predictions():
    X = np.random.rand(10, 3)
    y = np.zeros(10).astype(int)

    clf = NumericRandomForestClassifier(n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert np.all(preds == 0)
    
    
class MockTree:
    def __init__(self, label):
        self.label = label
    def predict(self, X):
        return np.full(len(X), self.label)

def test_majority_voting():
    X = np.random.rand(5, 3)
    clf = NumericRandomForestClassifier(n_estimators=3)
    clf.estimators = [MockTree(0), MockTree(1), MockTree(1)]
    
    preds = clf.predict(X)
    assert np.all(preds == 1)


def test_minimal_input():
    X = np.array([[0.1, 0.2]])
    y = np.array([1])
    
    clf = NumericRandomForestClassifier(n_estimators=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert preds.shape == (1,)
