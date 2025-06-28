from abc import ABC, abstractmethod
from typing import Optional


class BaseEstimator(ABC):
    """
    Base class for all estimators (e.g., regression, classification, clustering).
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fits the model to the training data.
        Parameters:
        - X: ndarray of shape (n_samples, n_features)
        - y: ndarray of shape (n_samples,) or None for clustering
        """
        pass


class BaseRegressor(BaseEstimator):
    """
    Interface for regression models.
    """

    @abstractmethod
    def predict(self, X):
        """
        Returns continuous predictions.
        """
        pass


class BaseClassifier(BaseEstimator):
    """
    Interface for classification models.
    """

    @abstractmethod
    def predict(self, X):
        """
        Returns class predictions.
        """
        pass


class BaseClusterer(BaseEstimator):
    """
    Interface for clustering algorithms.
    """

    @abstractmethod
    def predict(self, X):
        """
        Returns cluster labels for the input data.
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Convenience method: first fit, then predict.
        """
        self.fit(X)
        return self.predict(X)
    

class BaseProjector(BaseEstimator):
    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class TreeNode:
    def __init__(
        self, 
        threshold: Optional[float], 
        feature: Optional[int], 
        decision: Optional[int], 
        left: Optional["TreeNode"] = None, 
        right: Optional["TreeNode"] = None,
    ):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.decision = decision