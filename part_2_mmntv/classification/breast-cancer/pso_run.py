import numpy as np
from knn import KNNClassifier, DwKNNClassifier
from sklearn.metrics import accuracy_score as accuracy
from pso import GBestPSO, LBestPSO


def gbest_pso_run(x_train, y_train, x_test, y_test, x_verif, y_verif, k):
    #Run PSO to find best weights
    N_init_pop = 50

    _, nFeats = np.shape(x_train)
    weight_pso = GBestPSO(nFeats, N_init_pop)
    pos = weight_pso.get_positions()
    pbest = weight_pso.get_pbest()
    pbest_metric_array = np.empty(N_init_pop)
    pos_metric_array = np.empty(N_init_pop)

    #Set pbest metrics
    for i in range(len(pbest)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, pbest[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, pbest[i])

        #Method 1
        clf = KNNClassifier(scaled_x_train, y_train, k)
        neighbors = clf.find_all_neighbors(scaled_x_verif)
        nbh_ent = clf.find_neighborhood_entropy(neighbors)
        pbest_metric_array[i] = nbh_ent
    
    weight_pso.set_pbest_fitness(pbest_metric_array)

    #Set pos metrics
    for i in range(len(pbest)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, pos[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, pos[i])

        #Method 1
        clf = KNNClassifier(scaled_x_train, y_train, k)
        neighbors = clf.find_all_neighbors(scaled_x_verif)
        nbh_ent = clf.find_neighborhood_entropy(neighbors)
        pos_metric_array[i] = nbh_ent

    weight_pso.set_p_fitness(pos_metric_array)

    #Set initial gbest.
    weight_pso.set_init_best(pos_metric_array)

    count = 0
    while (count < 50):
        count += 1
        weight_pso.optimize()

        #get_population
        weight_pop = weight_pso.get_positions()
        metric_array = np.empty(N_init_pop)
    
        #evaluate and set fitness
        for i in range(len(weight_pop)):
            #Scale input data
            scaled_x_train = np.multiply(x_train, weight_pop[i])
            #Scale verificaion data
            scaled_x_verif = np.multiply(x_verif, weight_pop[i])
        
            #Method 1
            clf = KNNClassifier(scaled_x_train, y_train, k)
            neighbors = clf.find_all_neighbors(scaled_x_verif)
            nbh_ent = clf.find_neighborhood_entropy(neighbors)
            metric_array[i] = nbh_ent

        weight_pso.set_p_fitness(metric_array)
        weight_pso.set_best(metric_array)

        #get_best_sol
        best_metric = weight_pso.get_gbest_fit()

    best_weights = weight_pso.get_gbest()

    # Concatenate training and verification sets.
    x_train = np.concatenate((x_train, x_verif), axis = 0)
    y_train = np.concatenate([y_train, y_verif])

    # Print the results of KNN.
    clf = KNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    mse_iter = accuracy(y_test, y_pred)
    print("gbest-pso,knn,", k,",", mse_iter)

    # Print the results of KNN.
    clf = DwKNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    mse_iter = accuracy(y_test, y_pred)
    print("gbest-pso,dknn,", k,",", mse_iter)


def lbest_pso_run(x_train, y_train, x_test, y_test, x_verif, y_verif, k):
    #Run PSO to find best weights
    N_init_pop = 50

    _, nFeats = np.shape(x_train)
    weight_pso = LBestPSO(nFeats, N_init_pop)
    pos = weight_pso.get_positions()
    pbest = weight_pso.get_pbest()
    pbest_metric_array = np.empty(N_init_pop)
    pos_metric_array = np.empty(N_init_pop)

    #Set pbest metrics
    for i in range(len(pbest)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, pbest[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, pbest[i])

        #Method 1
        clf = KNNClassifier(scaled_x_train, y_train, k)
        neighbors = clf.find_all_neighbors(scaled_x_verif)
        nbh_ent = clf.find_neighborhood_entropy(neighbors)
        pbest_metric_array[i] = nbh_ent
    
    weight_pso.set_pbest_fitness(pbest_metric_array)

    #Set pos metrics
    for i in range(len(pbest)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, pos[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, pos[i])

        #Method 1
        clf = KNNClassifier(scaled_x_train, y_train, k)
        neighbors = clf.find_all_neighbors(scaled_x_verif)
        nbh_ent = clf.find_neighborhood_entropy(neighbors)
        pos_metric_array[i] = nbh_ent

    weight_pso.set_p_fitness(pos_metric_array)

    #Set initial gbest.
    weight_pso.set_init_best(pos_metric_array)

    count = 0
    while (count < 50):
        count += 1
        weight_pso.optimize()

        #get_population
        weight_pop = weight_pso.get_positions()
        metric_array = np.empty(N_init_pop)
    
        #evaluate and set fitness
        for i in range(len(weight_pop)):
            #Scale input data
            scaled_x_train = np.multiply(x_train, weight_pop[i])
            #Scale verificaion data
            scaled_x_verif = np.multiply(x_verif, weight_pop[i])
        
            #Method 1
            clf = KNNClassifier(scaled_x_train, y_train, k)
            neighbors = clf.find_all_neighbors(scaled_x_verif)
            nbh_ent = clf.find_neighborhood_entropy(neighbors)
            metric_array[i] = nbh_ent

        weight_pso.set_p_fitness(metric_array)
        weight_pso.set_best(metric_array)

        #get_best_sol
        best_metric = weight_pso.get_gbest_fit()

    best_weights = weight_pso.get_gbest()
   
    # Concatenate training and verification sets.
    x_train = np.concatenate((x_train, x_verif), axis = 0)
    y_train = np.concatenate([y_train, y_verif])

    # Print the results of KNN.
    clf = KNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    mse_iter = accuracy(y_test, y_pred)
    print("lbest-pso,knn,", k, ",", mse_iter)

    # Print the results of KNN.
    clf = DwKNNClassifier(np.multiply(x_train, best_weights), y_train, k)
    y_pred = clf.predict(np.multiply(x_test, best_weights))
    mse_iter = accuracy(y_test, y_pred)
    print("lbest-pso,dknn,", k, ",", mse_iter)

