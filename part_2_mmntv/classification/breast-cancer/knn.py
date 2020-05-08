import numpy as np
from numpy import linalg as LA

"""
KNN Regressor.
"""
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

"""
Inverse distance-weighted KNN regressor.
"""
class DwKNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []
            match_idx = -1 #Index of training sample if dist = 0.
            
            for i in range(len(self.X)):
                xi_dist = LA.norm(x - self.X[i])
                if xi_dist == 0:
                    match_idx = i
                    break
                else:
                    nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist
            
            if match_idx != -1:
                y = self.y[match_idx]
            else:
                sorted_dist_idx = np.argsort(nbrs_dist)
                k_idx = sorted_dist_idx[:self.k]
                weights = []

                y = 0.0
                for j in k_idx:
                    sample_wt = 1/nbrs_dist[j]
                    weights.append(sample_wt)
                    y += self.y[j] * sample_wt

                #y = y/(self.k)
                y = y/sum(weights)
    
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


"""
Inverse distance-weighted KNN Classifier.
"""
class DwKNNClassifier:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []
            match_idx = -1 #Index of training sample if dist = 0.
            
            for i in range(len(self.X)):
                xi_dist = LA.norm(x - self.X[i])
                if xi_dist == 0:
                    match_idx = i
                    break
                else:
                    nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist
            
            if match_idx != -1:
                y = self.y[match_idx]

            else:
                sorted_dist_idx = np.argsort(nbrs_dist)
                k_idx = sorted_dist_idx[:self.k]

                pred_dict = {}
                for j in k_idx:
                    sample_wt = 1/nbrs_dist[j]

                    if (self.y[j] in pred_dict):
                        pred_dict[self.y[j]] += sample_wt

                    else:
                        pred_dict[self.y[j]] = sample_wt

                y = max(pred_dict)
    
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

    def find_neighborhood_entropy(self, neighbors):
        entropies = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]

            (values,counts) = np.unique(y,return_counts=True)

            ent = 0
            for i in range(len(counts)):
                pi = counts[i]/self.k
                ent += (-pi * np.log(pi))

            entropies.append(ent)
        return(sum(entropies)/len(entropies))

    def find_neighborhood_std(self, neighbors):
        variances = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]
            var = np.var(y, ddof = 1)
            variances.append(var)
        return(np.sqrt(sum(variances)/len(variances)))

"""
KNN classifier.
"""
class KNNClassifier:
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
            y_k = []
            for i in range(0, self.k):
                y_k.append(self.y[k_idx[i]])
            # print(y_k)
            (values, counts) = np.unique(y_k,return_counts=True)
            # print('Values = ', values)
            # print('Counts = ', counts)
            y_idx = np.argmax(counts)
            y = values[y_idx]
            y_pred.append(y)
            # print(y)
            # print()

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

    def find_neighborhood_entropy(self, neighbors):
        entropies = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]

            (values,counts) = np.unique(y,return_counts=True)

            ent = 0
            for i in range(len(counts)):
                pi = counts[i]/self.k
                ent += (-pi * np.log(pi))

            entropies.append(ent)
        return(sum(entropies)/len(entropies))

    def find_neighborhood_std(self, neighbors):
        variances = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]
            var = np.var(y, ddof = 1)
            variances.append(var)
        return(np.sqrt(sum(variances)/len(variances)))


