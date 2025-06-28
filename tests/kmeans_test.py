import numpy as np
from ml_algorithms.unsupervised import KMeans

def test_kmeans_basic_clustering():
    X = np.vstack([
        np.random.randn(50, 2) * 0.5 + np.array([0, 0]),
        np.random.randn(50, 2) * 0.5 + np.array([5, 5])
    ])

    kmeans = KMeans(n_clusters=2, max_iter=100)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    assert len(labels) == len(X)

    unique, counts = np.unique(labels, return_counts=True)
    assert len(unique) == 2
    assert np.all(counts > 10)

def test_kmeans_centroid_shape():
    X = np.random.rand(100, 3)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    assert kmeans.centroids.shape == (4, 3)

def test_kmeans_predict_shape():
    X_train = np.random.rand(30, 2)
    X_test = np.random.rand(10, 2)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    labels = kmeans.predict(X_test)

    assert labels.shape == (10,)

def test_kmeans_converges_fast():
    X = np.vstack([
        np.random.randn(20, 2) + np.array([1, 1]),
        np.random.randn(20, 2) + np.array([-1, -1])
    ])

    kmeans = KMeans(n_clusters=2, max_iter=300, tol=1e-2)
    kmeans.fit(X)

    assert len(kmeans._historical_centroids) <= 300 * 2

def test_kmeans_empty_cluster_reinit():
    X = np.array([[0, 0], [0, 0], [10, 10]])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    assert set(labels).issubset({0, 1, 2})
    assert len(kmeans.centroids) == 3
