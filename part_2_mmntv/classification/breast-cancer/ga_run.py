import numpy as np
from knn import KNNClassifier, DwKNNClassifier
from ga import GeneticAlgorithm
from sklearn.metrics import accuracy_score as accuracy

def ga_run(x_train, y_train, x_test, y_test, x_verif, y_verif, k):
    # Run GA to find best weights.
    N_init_pop = 50
    N_crossover = 50
    N_selection = 20
    improv_thresh = 1e-3

    _, nFeats = np.shape(x_train)
    weight_ga = GeneticAlgorithm(nFeats, N_init_pop, mu = 0.1)
    weight_pop = weight_ga.get_population()
    metric_array = np.empty(N_init_pop)

    # Create the initial population.
    for i in range(len(weight_pop)):
        # Scale input data
        scaled_x_train = np.multiply(x_train, weight_pop[i])
        # Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, weight_pop[i])

        # Regressor.
        clf = KNNClassifier(scaled_x_train, y_train, k)
        neighbors = clf.find_all_neighbors(scaled_x_verif)
        nbh_ent = clf.find_neighborhood_entropy(neighbors)
        metric_array[i] = nbh_ent

    # Update fitness in GA object.
    weight_ga.set_fitness(metric_array)
    weight_ga.selection(N_selection)
    new_best_metric = 2.5

    # while (best_metric - new_best_metric) > improv_thresh:
    count = 0
    while (count < 20):
        count += 1
        best_metric = new_best_metric

        # Crossover.
        weight_ga.crossover(N_crossover)
    
        # Get new population.
        weight_pop = weight_ga.get_population()
        metric_array = np.empty(N_crossover)
    
        # Evaluate and set fitness.
        for i in range(len(weight_pop)):
            # Scale input data
            scaled_x_train = np.multiply(x_train, weight_pop[i])
            # Scale verificaion data
            scaled_x_verif = np.multiply(x_verif, weight_pop[i])
        
            # Regressor.
            clf = KNNClassifier(scaled_x_train, y_train, k)
            neighbors = clf.find_all_neighbors(scaled_x_verif)
            nbh_ent = clf.find_neighborhood_entropy(neighbors)
            metric_array[i] = nbh_ent

        # Update fitness in GA object
        weight_ga.set_fitness(metric_array)
        # get_best_sol
        best_weights, new_best_metric = weight_ga.best_sol()
        #print("Metric of this iteration are: ", new_best_metric)
        weight_ga.selection(N_selection)

    # print("Best weights = ", best_weights, "\tBest metric = ", new_best_metric)

    # Test with scaling after GA

    # Concatenate training and verification sets.
    x_train = np.concatenate((x_train, x_verif), axis = 0)
    y_train = np.concatenate([y_train, y_verif])

    # Print the results of KNN.
    clf = KNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    acc = accuracy(y_test, y_pred)
    print("ga,knn,", k, ",", acc)

    # Print the results of KNN.
    clf = DwKNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    mse_iter = accuracy(y_test, y_pred)
    print("ga,dknn,", k,",", mse_iter)
