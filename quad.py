''' author: samtenka
    change: 2019-09-30
    create: 2019-09-30
    descrp: takeuchi on toy landscapes
'''

import numpy as np
import scipy.linalg
from landscape import PointedLandscape

class QuadraticGauss(PointedLandscape): 
    def __init__(self, dim, hess=None, cvar=None):
        self.dim = dim
        self.hess = np.eye(self.dim) if hess is None else hess
        self.cvar = np.eye(self.dim) if cvar is None else cvar 
        self.root_cvar = scipy.linalg.sqrtm(self.cvar)

        self.reset_weights()

    def reset_weights(self, offset=True):
        self.weights = np.zeros(self.dim, dtype=np.float32) 
        if offset:
            self.weights[0] = 1.0

    def update_weights(self, displacement):
        self.weights += displacement

    def sample_data(self, nb_points):
        return np.random.randn(nb_points, self.dim)

    def loss_at(self, data): 
        signal = 0.5 * np.dot(np.matmul(self.hess, self.weights), self.weights)
        noise = (0.0 if data is None else
                 np.dot(np.matmul(self.root_cvar, np.mean(data, axis=0)), self.weights))
        return signal + noise

    def grad_at(self, data): 
        signal = np.matmul(self.hess, self.weights)
        noise = (0.0 if data is None else
                 np.matmul(self.root_cvar, np.mean(data, axis=0)))
        return signal + noise

    def hess_at(self, data): 
        return self.hess

    def jerk_at(self, data): 
        return np.zeros((self.dim, self.dim, self.dim), dtype=np.float32)
