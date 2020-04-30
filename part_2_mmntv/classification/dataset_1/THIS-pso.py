import numpy as np
from knn import KNNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ga import GeneticAlgorithm
from pso import GBestPSO
from sklearn.datasets import load_iris, load_digits, load_wine
import data_api as da

fdct = {1: da.AnuranCallsFamily(),\
        2: da.AnuranCallsGenus(),\
        3: da.AnuranCallsSpecies(),\
        4: da.AuditRisk(),\
        5: da.Avila(),\
        6: da.BankNoteAuth(),\
        7: da.BloodTransfusion(),\
        8: da.BreastCancer(),\
        9: da.BreastTissue(),\
        10: da.BurstHeaderPacket(),\
        11: da.CSection(),\
        12: da.CardioOtgMorph(),\
        13: da.CardioOtgFetal(),\
        14: da.DiabeticRetino(),\
        15: da.Ecoli(),\
        16: da.Electrical(),\
        17: da.EEGEye(),\
        18: da.Glass(),\
        19: da.Haberman(),\
        20: da.HTRU2(),\
        21: da.ILPD(),\
        22: da.Immunotherapy()}

for i in range (22, 23):
    print("i = ", i)
    dataset = fdct[i]
    X, y = dataset.Data()
    _, nFeats = np.shape(X)
    #if (nFeats > 15) or (_ > 4000):
    #    continue
    print("Number of samples: ",_, "Number of features: ", nFeats)
    #print("X :")
    #print(X)
    #print("y :")
    #print(y)
    print("Splitting training and test sets:")

    #Test without scaling
    print("Testing without scaling")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_train, x_verif, y_train, y_verif = train_test_split(x_train, y_train, test_size=0.33)

    clf = KNNClassifier(x_train, y_train, 5)
    y_pred = clf.predict(x_test)
    print("Accuracy = ", accuracy_score(y_test, y_pred))

    #Run GA to find best weights
    N_init_pop = 30

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
        reg = KNNClassifier(scaled_x_train, y_train, 5)
        neighbors = reg.find_all_neighbors(scaled_x_verif)
        nbh_std = reg.find_neighborhood_entropy(neighbors)
        pbest_metric_array[i] = nbh_std
    
    weight_pso.set_pbest_fitness(pbest_metric_array)

    #Set pos metrics
    for i in range(len(pbest)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, pos[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, pos[i])

        #Method 1
        reg = KNNClassifier(scaled_x_train, y_train, 5)
        neighbors = reg.find_all_neighbors(scaled_x_verif)
        nbh_std = reg.find_neighborhood_entropy(neighbors)
        pos_metric_array[i] = nbh_std

    weight_pso.set_p_fitness(pos_metric_array)

    #Set initial gbest.
    weight_pso.set_init_best(pos_metric_array)

    count = 0
    while (count < 100):
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
            reg = KNNClassifier(scaled_x_train, y_train, 5)
            neighbors = reg.find_all_neighbors(scaled_x_verif)
            nbh_std = reg.find_neighborhood_entropy(neighbors)
            metric_array[i] = nbh_std

        weight_pso.set_p_fitness(metric_array)
        weight_pso.set_best(metric_array)

        #get_best_sol
        best_metric = weight_pso.get_gbest_fit()
        print("Metric of this iteration are: ", best_metric, count)

    best_weights = weight_pso.get_gbest()
    print("Best weights = ", best_weights, "\tBest metric = ", best_metric)
    
    #Test with scaling after GA
    print("Testing with scaling using weights from GA")
    print(np.multiply(x_train, best_weights))
    reg = KNNClassifier(np.multiply(x_train, best_weights), y_train, 5)
    y_pred = reg.predict(np.multiply(x_test, best_weights))

    print(y_test)
    print(y_pred)

    print("Accuracy = ", accuracy_score(y_test, y_pred))
    print("-----------------------------------------------")

"""
#Test with scaling after exhaustive search
print("Testing with scaling using weights from exhaustive search")
best_weights = np.array([0.1, 0.8, 0.0])
reg = KNNRegressor(np.multiply(x_train, best_weights), y_train, 5)
y_pred = reg.predict(np.multiply(x_test, best_weights))
print("MSE = ", mean_squared_error(y_test, y_pred))

"""
#print("Best weights = ", lin_combos[best_idx], "\tBest metric:", metric_array[best_idx])
