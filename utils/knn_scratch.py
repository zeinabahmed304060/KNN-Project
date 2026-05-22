import numpy as np
from collections import Counter

class KNN_From_Scratch:
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_idx = np.argsort(distances)[:self.k]
        labels = self.y_train[k_idx]
        return Counter(labels).most_common(1)[0][0]