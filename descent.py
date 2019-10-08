''' author: samtenka
    change: 2019-10-08
    create: 2019-10-07
    descrp: descend on a stochastic landscape
'''

import numpy as np
from landscape import PointedLandscape
import tqdm

def gd_test(T, eta, N, landscape, I=1): 
    rtrns = []
    for i in tqdm.tqdm(range(I)):
        train = landscape.sample_data(N) 

        landscape.reset_weights()
        for t in range(T): 
            g = landscape.grad_at(train)
            landscape.update_weights( -eta * g )

        rtrns.append(landscape.loss_at(None))
    return np.mean(rtrns), np.std(rtrns)

def gdt_test(T, eta, N, landscape, I=1): 
    rtrns = []
    for i in tqdm.tqdm(range(I)):
        train = landscape.sample_data(N) 
        trainA= train[:N//2]
        trainB= train[N//2:]
    
        landscape.reset_weights()
        for t in range(T): 
            g, gA, gB = (landscape.grad_at(dset) for dset in (train, trainA, trainB))
            h, hA     = (landscape.hess_at(dset) for dset in (train, trainA        ))
            j,        = (landscape.jerk_at(dset) for dset in (train,               ))
            hinv = np.linalg.inv(h) 
    
            X = np.tensordot(hinv, j, axes=((0,), (0,)))
            Y = np.tensordot(X, hinv, axes=((0,), (0,)))
            treg = (
                  np.matmul(np.matmul(hA, hinv), gA-gB)
                - np.sum(np.multiply(Y, np.outer(gA, gA-gB)))
            )
    
            landscape.update_weights( -eta * (g + treg/N) )
    
        rtrns.append(landscape.loss_at(None))
    return np.mean(rtrns), np.std(rtrns)
    
