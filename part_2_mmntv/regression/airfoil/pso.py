import numpy as np

class GBestPSO:
    def __init__(self,\
                 n_dims,\
                 n_init,\
                 c1 = 0.5,\
                 c2 = 0.5):

        self.n_dims = n_dims
        self.n_init = n_init

        self.p = np.random.rand(n_init, n_dims) * 10
        self.v = np.zeros((n_init, n_dims))
        
        self.c1 = c1
        self.c2 = c2

        self.pbest = np.random.rand(n_init, n_dims) * 50 + 50
        #self.gbestIdx = np.argmin(objective(p))
        #self.gbest = np.array([p[gbestIdx]])
        
        
        self.gbestIdx = None
        self.gbest = None
        self.gbestFit = None

        self.p_objectives = None
        self.pbest_objectives = None

    def get_positions(self):
        return self.p

    def get_pbest(self):
        return self.pbest

    def set_p_fitness(self, objective_vals):
        self.p_objectives = objective_vals

    def set_pbest_fitness(self, objective_vals):
        self.pbest_objectives = objective_vals

    def get_gbest_fit(self):
        return self.gbestFit

    def get_gbest(self):
        return self.gbest
    # Assume that fitness is calculated within the main function.
    def optimize(self):    
        mask = self.p_objectives <= self.pbest_objectives

        for i in range(self.n_init):
            if mask[i]:
                self.pbest[i] = self.p[i]
                self.pbest_objectives[i] = self.p_objectives[i]

        self.v = self.v +\
                 self.c1 * np.random.rand(self.n_init, self.n_dims) *\
                                         (self.pbest - self.p) +\
                 self.c2 * np.random.rand(self.n_init, self.n_dims) *\
                                         (self.gbest - self.p)
        self.p = self.p + self.v

    def set_best(self, objective_vals):
        newGbestIdx = np.argmin(objective_vals)
        newGbest = np.array([self.p[newGbestIdx]])
        newGbestFit = objective_vals[newGbestIdx]

        if newGbestFit < self.gbestFit:
            self.gbest = newGbest
            self.gbestFit = newGbestFit

    def set_init_best(self, objective_vals):
        gbestIdx = np.argmin(objective_vals)
        self.gbest = np.array([self.p[gbestIdx]])
        self.gbestFit = objective_vals[gbestIdx]

class LBestPSO:
    def __init__(self,\
                 n_dims,\
                 n_init,\
                 c1 = 0.5,\
                 c2 = 0.5):

        self.n_dims = n_dims
        self.n_init = n_init

        self.p = np.random.rand(n_init, n_dims) * 10
        self.v = np.zeros((n_init, n_dims))
        
        self.c1 = c1
        self.c2 = c2

        self.pbest = np.random.rand(n_init, n_dims) * 50 + 50
        #self.gbestIdx = np.argmin(objective(p))
        #self.gbest = np.array([p[gbestIdx]])
        
        
        self.gbestIdx = None
        self.gbest = None
        self.gbestFit = None

        self.p_objectives = None
        self.pbest_objectives = None

    def get_positions(self):
        return self.p

    def get_pbest(self):
        return self.pbest

    def set_p_fitness(self, objective_vals):
        self.p_objectives = objective_vals

    def set_pbest_fitness(self, objective_vals):
        self.pbest_objectives = objective_vals

    def get_gbest_fit(self):
        return self.gbestFit

    def get_gbest(self):
        return self.gbest
    # Assume that fitness is calculated within the main function.
    def optimize(self):    
        mask = self.p_objectives <= self.pbest_objectives

        for i in range(self.n_init):
            if mask[i]:
                self.pbest[i] = self.p[i]
                self.pbest_objectives[i] = self.p_objectives[i]

        # Neighbor pbest
        nbhPbest = np.copy(self.pbest)
        for dim in range(self.n_dims):
            nbhPbest[:, dim] = np.roll(nbhPbest[:, dim], -1)

        self.v = self.v +\
                 self.c1 * np.random.rand(self.n_init, self.n_dims) *\
                                         (self.pbest - self.p) +\
                 self.c2 * np.random.rand(self.n_init, self.n_dims) *\
                                         (nbhPbest - self.p)
        self.p = self.p + self.v

    def set_best(self, objective_vals):
        newGbestIdx = np.argmin(objective_vals)
        newGbest = np.array([self.p[newGbestIdx]])
        newGbestFit = objective_vals[newGbestIdx]

        if newGbestFit < self.gbestFit:
            self.gbest = newGbest
            self.gbestFit = newGbestFit

    def set_init_best(self, objective_vals):
        gbestIdx = np.argmin(objective_vals)
        self.gbest = np.array([self.p[gbestIdx]])
        self.gbestFit = objective_vals[gbestIdx]
