import numpy as np
from numpy import linalg as LA

class KNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            for j in k_idx:
                y += self.y[j]

            y = y/(self.k)
            y_pred.append(y)
        return y_pred

    def find_all_neighbors(self, X_test):
        neighbors = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            neighbors.append(sorted(k_idx))
        return neighbors

    def find_neighborhood_std(self, neighbors):
        variances = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]
            var = np.var(y, ddof = 1)
            variances.append(var)
        return(np.sqrt(sum(variances)/len(variances)))
