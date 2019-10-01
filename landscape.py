''' author: samtenka
    change: 2019-09-30
    create: 2019-09-30
    descrp: takeuchi on toy landscapes
'''

import numpy as np
from abc import ABC, abstractmethod

class PointedLandscape(ABC): 
    ''' Interface for stochastic loss landscape equipped with a current point in weightspace.
    '''
    @abstractmethod
    def sample_data(self, nb_points):
        ''' give array of shape (nb_points, ...) sufficient for
            computing loss stalks on corresponding data
        '''
        pass

    @abstractmethod
    def reset_weights(self):
        ''' reset internal weights by sampling from a fixed distribution
            must be called at __init__.
        '''
        pass

    @abstractmethod
    def update_weights(self, displacement):
        ''' add displacement to internal weights '''
        pass

    @abstractmethod
    def loss_at(self, data):  
        ''' give average loss on data (scalar) --- or test loss if data is None. '''
        pass

    @abstractmethod
    def grad_at(self, data): 
        ''' give average grad on data (vector) --- or test grad if data is None. '''
        pass

    @abstractmethod
    def hess_at(self, data): 
        ''' give average hess on data (matrix) --- or test hess if data is None. '''
        pass

