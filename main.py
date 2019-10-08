''' author: samtenka
    change: 2019-09-30
    create: 2019-09-30
    descrp: takeuchi on toy landscapes
'''

import numpy as np
from quad import QuadraticGauss

D, N = 16, 8
LRATE = 0.01

cvar=np.diag([0.1]*(D//4) + [ 0.1]*(D//4) + [0.2]*(D//4) + [0.2]*(D//4))
hess=np.diag([2.0]*(D//4) + [ 1.0]*(D//4) + [2.0]*(D//4) + [2.0]*(D//4))

QG = QuadraticGauss(dim=D, hess=hess, cvar=cvar)
data = QG.sample_data(nb_points=N)

print('ORDINARY GD...')
for t in range(201):
    if t%20==0:
        print('step {:4d} \t train {:6.3f} \t test {:6.3f}'.format(
            t, QG.loss_at(data), QG.loss_at(None))
        )
    g = QG.grad_at(data)
    QG.update_weights(- LRATE * g)

