import numpy as np
import data_api as da
import multiprocessing
import time
from knn import KNNRegressor, DwKNNRegressor
from sklearn.model_selection import train_test_split
from ga import GeneticAlgorithm
from sklearn.metrics import mean_squared_error as skmse
from ga_run import ga_run
from pso_run import gbest_pso_run, lbest_pso_run
from sklearn.decomposition import PCA

# Load data.
data = da.EnergyHeat()
X, y = data.Data()
_, nFeats = np.shape(X)

# Values of parameter k to iterate over.
K_VALS = [3, 5, 7, 9, 11, 13, 15]

starttime = time.time()
# Repeat each trial 10 times.
for i in range (0, 10):
    x_train, x_test, y_train, y_test = train_test_split(X, y,\
                                                        test_size=0.2)

    """
    Try non-optimized methods.
    """
    # Vanilla KNN.
    for k in K_VALS:
        reg = KNNRegressor(x_train, y_train, k)
        y_pred = reg.predict(x_test)
        mse_iter = skmse(y_test, y_pred)
        print("xx,knn,", k,",", mse_iter)
    
    # Distance-weighted KNN.
    for k in K_VALS:
        reg = DwKNNRegressor(x_train, y_train, k)
        y_pred = reg.predict(x_test) 
        mse_iter = skmse(y_test, y_pred)
        print("xx,dknn,", k,",", mse_iter)

    """
    PCA with KNN.
    """
    pca = PCA(n_components = 4)
    pca.fit(x_train.copy())
    x_train_pca = pca.transform(x_train.copy())
    x_test_pca = pca.transform(x_test.copy())
    
    # PCA + Vanilla KNN.
    for k in K_VALS:
        reg = KNNRegressor(x_train_pca, y_train, k)
        y_pred = reg.predict(x_test_pca)
        mse_iter = skmse(y_test, y_pred)
        print("pca,knn,", k,",", mse_iter)
   
    # PCA + Distance-weighted KNN.
    for k in K_VALS:
        reg = DwKNNRegressor(x_train_pca, y_train, k)
        y_pred = reg.predict(x_test_pca) 
        mse_iter = skmse(y_test, y_pred)
        print("pca,dknn,", k,",", mse_iter)

    x_train, x_verif, y_train, y_verif = train_test_split(x_train,\
                                                          y_train,\
                                                          test_size=0.33)
    """
    GA-driven methods.
    """
    # Use different values of k.
    for k in K_VALS:
        # Run the GA based optimization.
        ga_run(x_train.copy(),\
               y_train.copy(),\
               x_test.copy(),\
               y_test.copy(),\
               x_verif.copy(),\
               y_verif.copy(),\
               k)


    """
    GBest_PSO-driven methods.
    """
    # Use different values of k.
    for k in K_VALS:
        # Run the GA based optimization.
        gbest_pso_run(x_train.copy(),\
                      y_train.copy(),\
                      x_test.copy(),\
                      y_test.copy(),\
                      x_verif.copy(),\
                      y_verif.copy(),\
                      k)


    """
    LBest_PSO-driven methods.
    """
    # Use different values of k.
    for k in K_VALS:
        # Run the GA based optimization.
        lbest_pso_run(x_train.copy(),\
                      y_train.copy(),\
                      x_test.copy(),\
                      y_test.copy(),\
                      x_verif.copy(),\
                      y_verif.copy(),\
                      k)

print('That took {} seconds'.format(time.time() - starttime))
