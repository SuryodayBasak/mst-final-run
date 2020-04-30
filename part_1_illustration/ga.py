import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, n_dims, N_init, mu = 0.001):
        self.n_dims = n_dims
        self.mu = mu
        #Initialize population
        self.population = np.random.rand(N_init, self.n_dims)
        self.fitness = None

    def crossover(self, N_cross):
        #Implement crossover before applying the solution
        new_pop = np.empty((N_cross, self.n_dims))
        for i in range(N_cross):
            parent_1 = self.population[random.randint(0, len(self.population)-1)]
            parent_2 = self.population[random.randint(0, len(self.population)-1)]
            child = (parent_1 + parent_2)/2
            new_pop[i] = self.mutate(child)
        self.population = new_pop

    def mutate(self, gene):
        mutated_child = np.empty(self.n_dims)
        for i in range(self.n_dims):
            mutated_child[i] = gene[i] + (self.mu*np.random.uniform(-1, 1))
        return mutated_child

    def best_sol(self):
        #return w, fitness
        fitness_args = np.argsort(self.fitness)
        return self.population[fitness_args[0]], self.fitness[fitness_args[0]]

    def selection(self, N_select):
        #Argsort first
        fitness_args = np.argsort(self.fitness)
        self.population = self.population.take(fitness_args[0:N_select], axis=0)

    def get_population(self):
        return self.population

    def set_fitness(self, fitness):
        self.fitness = fitness
