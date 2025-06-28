import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ml_algorithms.unsupervised import PCAWithScaler

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(100, 10) * 2 + 3

def test_fit_sets_components_and_stats(sample_data):
    pca = PCAWithScaler(n_components=5)
    pca.fit(sample_data)

    assert pca._components is not None
    assert pca._mean is not None
    assert pca._std is not None
    assert pca._components.shape == (10, 5)
    assert pca._mean.shape == (1, 10)
    assert pca._std.shape == (1, 10)

def test_transform_shape(sample_data):
    pca = PCAWithScaler(n_components=3)
    pca.fit(sample_data)
    transformed = pca.transform(sample_data)

    assert transformed.shape == (100, 3)

def test_reverse_transform_recovers_original_shape(sample_data):
    pca = PCAWithScaler(n_components=5)
    pca.fit(sample_data)
    transformed = pca.transform(sample_data)
    recovered = pca.reverse_transform(transformed)

    assert recovered.shape == sample_data.shape

def test_fit_transform_consistency(sample_data):
    pca = PCAWithScaler(n_components=4)
    pca.fit(sample_data)
    manual = pca.transform(sample_data)
    combined = pca.fit_transform(sample_data)

    assert_array_almost_equal(manual, combined)

def test_reverse_transform_accuracy(sample_data):
    pca = PCAWithScaler(n_components=10)
    pca.fit(sample_data)
    transformed = pca.transform(sample_data)
    recovered = pca.reverse_transform(transformed)

    assert_array_almost_equal(recovered, sample_data, decimal=5)

def test_transform_without_fit_raises():
    pca = PCAWithScaler(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(np.random.randn(10, 2))

def test_reverse_transform_without_fit_raises():
    pca = PCAWithScaler(n_components=2)
    with pytest.raises(RuntimeError):
        pca.reverse_transform(np.random.randn(10, 2))
