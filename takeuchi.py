''' author: samtenka
    change: 2019-10-08
    create: 2019-10-07
    descrp: takeuchi regularization for a stochastic landscape
'''

import numpy as np
from landscape import PointedLandscape
from descent import gd_test, gdt_test
from quad import QuadraticGauss 

D = 8
cvar=np.diag([ 0.010]*(D//4) + [ 0.100]*(D//4) + [ 1.000]*(D//4) + [10.00 ]*(D//4))
hess=np.diag([ 1.000]*(D//4) + [ 0.100]*(D//4) + [ 0.010]*(D//4) + [ 0.001]*(D//4))
QG = QuadraticGauss(dim=D, hess=hess, cvar=cvar)

I = 1000
mean, std = gd_test(T=1000, N=5, eta=0.1, landscape=QG, I=I)
print('test {:.2f} ({:.2f})'.format(mean, std/I**0.5))
mean, std = gdt_test(T=1000, N=5, eta=0.1, landscape=QG, I=I)
print('test {:.2f} ({:.2f})'.format(mean, std/I**0.5))
