import numpy as np
from knn import KNNRegressor
from sklearn.metrics import mean_squared_error
from ga import GeneticAlgorithm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def return_y(x, y, noise):
    #return 10*np.sin(0.05*np.pi*x)*np.sin(0.1*np.pi*y) + np.random.normal(0, noise)
    #return (2*x) + (5*y) + np.random.normal(0, noise)
    return (5*x) + np.random.normal(0, noise)

"""
SEGMENT 1: Create data and plots.
"""
# Create a figure with two subplots.
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
bx = fig.add_subplot(122, projection='3d')

# Subplot 1.
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
bx.set_xlabel('X1')
bx.set_ylabel('X2')
bx.set_zlabel('y')


mean_1 = (5, 2)
cov_1 = [[1, 0.8], [0.8, 1]]

mean_2 = (0, -1)
cov_2 = [[1, 0.1], [0.1, 1]]

noise = 1.0
N = 100
n_div = 50

X_1 = np.random.multivariate_normal(mean_1, cov_1, n_div)
X_2 = np.random.multivariate_normal(mean_2, cov_2, N-n_div)

X = np.concatenate((X_1, X_2), axis = 0) 
y = []

for sample in X:
    y_ = return_y(sample[0], sample[1], noise)
    y.append(y_)

y = np.array(y)

# Add a column of random numbers to the data

rand_col = 100*np.random.rand(len(X), 1)
#X = np.append(X, rand_col, axis = 1)

x_train = X.copy()
y_train = y.copy()
ax.scatter(X_1, X_2, y)
"""
Segment 2: Generate verification data.
"""

#Generate verification data
X_1 = np.random.multivariate_normal(mean_1, cov_1, 100)
X_2 = np.random.multivariate_normal(mean_2, cov_2, 100)
x_verif = np.concatenate((X_1, X_2), axis = 0)
#Add a column of random numbers to the data
#rand_col_verif = 100*np.random.rand(len(x_verif))
#x_verif = np.insert(x_verif, 2, rand_col_verif, axis=1)


"""
Segment 3: GA, I guess.
"""

#Run GA to find best weights
N_init_pop = 50
N_crossover = 100
N_selection = 50
improv_thresh = 1e-3

print("Step 1.")

# weight_ga = GeneticAlgorithm(3, N_init_pop, mu = 0.1)
weight_ga = GeneticAlgorithm(2, N_init_pop, mu = 0.1)
weight_pop = weight_ga.get_population()
metric_array = np.empty(N_init_pop)
for i in range(len(weight_pop)):
    #Scale input data
    scaled_x_train = np.multiply(x_train, weight_pop[i])
    #Scale verificaion data
    scaled_x_verif = np.multiply(x_verif, weight_pop[i])

    #Method 1
    reg = KNNRegressor(scaled_x_train, y_train, 5)
    neighbors = reg.find_all_neighbors(scaled_x_verif)
    nbh_std = reg.find_neighborhood_std(neighbors)
    metric_array[i] = nbh_std

#Update fitness in GA object
weight_ga.set_fitness(metric_array)
weight_ga.selection(N_selection)
new_best_metric = 2.5

#while (best_metric - new_best_metric) > improv_thresh:
count = 0
print("About to start GA.")
while (count < 15):
    count += 1
    best_metric = new_best_metric
    #crossover
    weight_ga.crossover(N_crossover)
    
    #get_population
    weight_pop = weight_ga.get_population()
    metric_array = np.empty(N_crossover)
    
    #evaluate and set fitness
    for i in range(len(weight_pop)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, weight_pop[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, weight_pop[i])
        
        #Method 1
        reg = KNNRegressor(scaled_x_train, y_train, 5)
        neighbors = reg.find_all_neighbors(scaled_x_verif)
        nbh_std = reg.find_neighborhood_std(neighbors)
        metric_array[i] = nbh_std

    #selection
    #Update fitness in GA object
    weight_ga.set_fitness(metric_array)
    #get_best_sol
    best_weights, new_best_metric = weight_ga.best_sol()
    weight_ga.selection(N_selection)

    print("Best weights = ", best_weights, "\tBest metric = ", new_best_metric)

scaled_x_train = np.multiply(x_train, best_weights)

bx.scatter(x_train[:, 0], x_train[:, 1], y)
ax.set_title("Without scaling")
bx.set_title("With scaling")
plt.show()

"""
Testing on synthetic data.
"""

X_test_1 = np.random.multivariate_normal(mean_1, cov_1, n_div)
X_test_2 = np.random.multivariate_normal(mean_2, cov_2, N - n_div)

X_test = np.concatenate((X_test_1, X_test_2), axis = 0) 
y_test = []

for sample in X_test:
    y_ = return_y(sample[0], sample[1], noise)
    y_test.append(y_)

y_test = np.array(y_test)

# Test on unscaled data.
reg = KNNRegressor(X, y, 5)
y_pred = reg.predict(X_test)
print(mean_squared_error(y_test, y_pred))

# Test on scaled data.
reg = KNNRegressor(scaled_x_train, y, 5)
y_pred = reg.predict(np.multiply(X_test, best_weights))
print(mean_squared_error(y_test, y_pred))

