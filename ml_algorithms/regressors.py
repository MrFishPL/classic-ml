from ml_algorithms.base import BaseRegressor, TreeNode
from typing import Optional
import numpy as np

class LinearRegression(BaseRegressor):    
    W: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None 
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        
        # (batch_size, dim) - (1, dim)
        X_centered = X - X_mean[np.newaxis, ...]
        y_centered = y - y_mean[np.newaxis, ...]
        
        self.W = np.linalg.pinv(X_centered.T @ X_centered) @ X_centered.T @ y_centered
        self.B = y_mean - X_mean @ self.W
    
    def predict(self, X: np.ndarray):
        if self.W is None or self.B is None:
            raise Exception("Fit model first.")
        
        return X @ self.W + self.B