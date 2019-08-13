import random
import math
import numpy as np

def getNextAction(Q, state, epsilon):
    Qpos = Q[1].feed(state)
    Qneg = Q[-1].feed(state)
    u = 1
    if Qneg>Qpos:
        u = -1
    else:
        u = 1
    if epsilon<random.uniform(0,1):
        u = np.sign(random.uniform(-1,1))
    return u

class RBFFeature:
    def __init__(self, deltax, deltav, i, j, vm):
        self.cx = 0+i*deltax
        self.cv = -vm + j * deltav
        self.varx = math.sqrt(1/deltax)
        self.varv = math.sqrt(1/deltav)
    def calc(self,s):
        return math.exp(-( (s[0]-self.cx)**2/self.varx**2 + (s[1]-self.cv)**2/self.varv**2)/2 )

class RBF:
    def __init__(self, nx=5, nv=5, L=5, vm=5.0):
        deltax = L/(nx-1)
        deltav = 2*vm/(nv-1)
        self.features = []
        for i in range(nx):
            for j in range(nv):
                self.features.append(RBFFeature(deltax,deltav,i,j,vm))
        self.weights = np.random.uniform(low=-1, high=1, size=(nx*nv,))

    def feed(self, state):
        input_features = []
        for i in range(len(self.features)):
            input_features.append(self.features[i].calc(state))
        input_features = np.asarray(input_features).reshape((len(self.features),1))
        res = self.weights.transpose().dot(input_features)
        return res

if __name__ == '__main__':
    learning_rate = 0.9

    g = 9.8
    h = 5
    nx = 5
    nv = 5
    L = 4
    vm = math.sqrt(2*g*h)

    Q = {-1:RBF(nx,nv,L,vm), 1:RBF(nx,nv,L,vm)}

    episode = 0
    epsilon0 = 0
    error = 1.0
    while error>0.001:
        episode+=1
        epsilon = epsilon0/episode

        st = [L/2,0]
        at = int(getNextAction(Q, st, epsilon))
