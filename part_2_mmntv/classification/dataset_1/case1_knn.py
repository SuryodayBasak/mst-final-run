import numpy as np
from knn import KNNRegressor
from sklearn.metrics import mean_squared_error as skmse
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_wine
import data_api as da

data = da.Mpg()
X, y = data.Data()

print(X, y)

K_VALS = [3, 5, 7, 9, 11, 13, 15]

# Use different values of k.
for k in K_VALS:
    dataset = DATASETTTT
    X, y = dataset.Data()
    _, nFeats = np.shape(X)
    mse_list = []    

    # Repeat each trial 10 times.
    for i in range (0, 10):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        reg = KNNRegressor(x_train, y_train, k)
        y_pred = reg.predict(x_test)
    
        iter_mse = skmse(y_test, y_pred)
        mse_list.append(iter_mse)
        print("MSE = ", iter_mse)



