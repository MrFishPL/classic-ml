from ml_algorithms.base import BaseProjector, BaseClusterer
import numpy as np


class PCAWithScaler(BaseProjector):
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components
        self._components = None
        self._mean = None
        self._std = None
    
    def fit(self, X: np.ndarray):
        self._mean = X.mean(axis=0)[np.newaxis, ...]
        self._std = X.std(axis=0)[np.newaxis, ...]
        
        X_scaled: np.ndarray = (X - self._mean)/self._std
        cov_matrix = np.cov(X_scaled, rowvar=False)
        eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
        
        sorting_indicies = eig_values.argsort()[::-1]
        sorted_vectors = eig_vectors[:, sorting_indicies]
        
        self._components = sorted_vectors[:, :self.n_components]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._components is None:
            raise RuntimeError("Fit model first.")

        X_scaled: np.ndarray = (X - self._mean)/self._std
        return X_scaled @ self._components
    
    def reverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        if self._components is None:
            raise RuntimeError("Fit model first.")

        X_scaled = X_transformed @ self._components.T
        X = X_scaled * self._std + self._mean
        return X


class KMeans(BaseClusterer):
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 0.0001):
        super().__init__()
        
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        
        self._historical_centroids = []
        self._historical_labels = []
    
    def fit(self, X: np.ndarray):
        clusters_indecies = np.random.choice(
            np.arange(X.shape[0]), size=self.n_clusters, replace=False
        )
        
        self.centroids = X[clusters_indecies, :]
        
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)        
            closest = np.argmin(dists, axis=1)
            self._historical_centroids.append(np.copy(self.centroids))
            self._historical_labels.append(np.copy(closest))
            
            curr_change = 0
            for i in range(self.n_clusters):
                mask = closest == i
                
                new_centroid = None
                if np.any(mask):    new_centroid = X[mask, :].mean(axis=0)
                else:               new_centroid = X[np.random.randint(0, X.shape[0])]
                    
                curr_change += np.abs(self.centroids[i] - new_centroid).sum()
                self.centroids[i] = new_centroid
            
            if curr_change < self.tol:
                break 
            
    def predict(self, X: np.ndarray):
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)        
        closest = np.argmin(dists, axis=1)
        
        return closest