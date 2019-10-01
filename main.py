''' author: samtenka
    change: 2019-09-30
    create: 2019-09-30
    descrp: takeuchi on toy landscapes
'''

import numpy as np
from quad import QuadraticGauss

D, N = 16, 8
LRATE = 0.1

cvar=np.diag([10.0]*(D//4) + [ 1.0]*(D//4) + [10.0]*(D//4) + [10.0]*(D//4))
hess=np.diag([10.0]*(D//4) + [ 1.0]*(D//4) + [10.0]*(D//4) + [10.0]*(D//4))

QG = QuadraticGauss(dim=D, hess=hess, cvar=cvar)
data = QG.sample_data(nb_points=N)

print('ORDINARY GD...')
for t in range(100):
    if t%10==0:
        print('step {} \t train {:0.3f} \t test {:0.3f}'.format(
            t, QG.loss_at(data), QG.loss_at(None))
        )
    g = QG.grad_at(data)
    QG.update_weights(- LRATE * g)

#print('GD W TAKEUCHI...')
#for t in range(100):
#    if t%10==0:
#        print('step {} \t train {:0.3f} \t test {:0.3f}'.format(
#            t, QG.loss_at(data), QG.loss_at(None))
#        )
#    g = QG.grad_at(data)
#    QG.update_weights(- LRATE * g)
