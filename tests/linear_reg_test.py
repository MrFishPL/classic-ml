from ml_algorithms.regressors import LinearRegression
import numpy as np

def test_linear_regression():
    X = np.random.randn(1000, 10)
    W = np.random.randn(10, 20)
    B = np.random.randn(20)
    y = X @ W + B

    reg = LinearRegression()

    reg.fit(X, y)

    X_new = np.random.randn(1000, 10)
    y_new = X_new @ W + B

    prediction = reg.predict(X_new)

    assert np.allclose(prediction, y_new, atol=1e-15)
    assert np.allclose(W, reg.W, atol=1e-15)
    assert np.allclose(B, reg.B, atol=1e-15)