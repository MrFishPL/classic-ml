from ml_algorithms.base import BaseClassifier, TreeNode
import numpy as np
from typing import Optional
from tqdm import tqdm

class NumericDecisionTreeClassifier(BaseClassifier):
    def __init__(
        self, 
        max_depth: Optional[int] = None, 
        min_samples_split: Optional[int] = 2, 
        draw_features: Optional[int] = None,
    ):
        super().__init__()
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.draw_features = draw_features
        
        self._X = None
        self._root = None
    
    def _gini(self, targets_subset: np.ndarray) -> float:
        labels = np.unique(targets_subset)
        
        probs = np.array([
            float((targets_subset == label).sum()) / float(targets_subset.size) 
            for label in labels
        ])
        
        return 1 - (probs**2).sum()
    
    def _get_split(self, X_subset, y_subset):
        best_feature = -1
        best_threshold = -1
        best_gini = float("inf")
        
        features_total = X_subset.shape[1]
        if self.draw_features is not None and self.draw_features < features_total:
            feature_indices = np.random.choice(
                features_total, self.draw_features, replace=False
            )
        else:
            feature_indices = np.arange(features_total)
        
        for i in feature_indices:
            dims_sorted = np.sort(X_subset[:, i])
            thresholds = (dims_sorted[1:] + dims_sorted[:-1]) / 2

            for thr in thresholds:
                mask = X_subset[:, i] < thr
                num_positives = mask.sum()
                
                if not(
                    num_positives > self.min_samples_split and 
                    mask.size - num_positives > self.min_samples_split
                ):
                    continue
                
                gini_left = self._gini(y_subset[mask])
                gini_right = self._gini(y_subset[~mask])
                
                wght = (float(mask.sum()) / len(mask))
                gini_score = wght * gini_left + (1 - wght) * gini_right

                if gini_score < best_gini:
                    best_feature = i
                    best_threshold = thr
                    best_gini = gini_score
        
        return best_feature, best_threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # For tests
        self._X = X
        
        self._root = TreeNode(threshold=None, feature=None, decision=None)
        stack = [(X, y, 0, self._root)]

        while stack:
            X_subset, y_subset, depth, node = stack.pop()

            if (
                len(np.unique(y_subset)) == 1 or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(y_subset) < self.min_samples_split
            ):
                node.decision = int(np.bincount(y_subset).argmax())
                continue

            feature, threshold = self._get_split(X_subset, y_subset)

            if feature == -1:
                node.decision = int(np.bincount(y_subset).argmax())
                continue

            node.threshold = threshold
            node.feature = feature
            node.decision = None

            mask = X_subset[:, feature] < threshold

            node.left = TreeNode(threshold=None, feature=None, decision=None)
            node.right = TreeNode(threshold=None, feature=None, decision=None)

            stack.append((X_subset[~mask], y_subset[~mask], depth + 1, node.right))
            stack.append((X_subset[mask], y_subset[mask], depth + 1, node.left))

    def _predict_one(self, x: np.ndarray) -> int:
        node = self._root
        while node.decision is None:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.decision

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])
    
    
class NumericRandomForestClassifier(BaseClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, n_estimators=100, show_progress=False):
        super().__init__()
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.show_progress = show_progress
        self.estimators = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimators = [
            NumericDecisionTreeClassifier( 
                self.max_depth, 
                self.min_samples_split, 
                int(np.ceil(np.sqrt(X.shape[-1])))
            ) for _ in range(self.n_estimators)
        ]
        
        dataset_len = X.shape[0]
        indices = np.random.choice(dataset_len, size=dataset_len, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        iterator = self.estimators
        if self.show_progress:
            iterator = tqdm(self.estimators, "Training...")
        
        for estimator in iterator:
            estimator.fit(X_bootstrap, y_bootstrap)
    
    def predict(self, X: np.ndarray):
        votes = np.array([estimator.predict(X) for estimator in self.estimators])
        votes_t = votes.T
        
        num_classes = np.max(votes) + 1
        one_hot = np.eye(num_classes)[votes_t]
        counts = one_hot.sum(axis=1)
        majority_votes = np.argmax(counts, axis=1)
        
        return majority_votes