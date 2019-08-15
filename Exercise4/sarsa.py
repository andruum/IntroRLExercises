import random
import math
import numpy as np
from carOnAMountain import CoM as sim
from plotter import Plotter

def getNextAction(Q, state, epsilon):
    Qpos = Q[1].feed(state)
    Qneg = Q[-1].feed(state)
    u = 1
    if Qneg>Qpos:
        u = -1
    else:
        u = 1
    if epsilon>random.uniform(0,1):
        u = np.sign(random.uniform(-1,1))
        # print("Explorer")
    return u

class RBFFeature:
    def __init__(self, deltax, deltav, i, j, vm):
        self.cx = 0+i*deltax
        self.cv = -vm + j * deltav
        self.varx = math.sqrt(3/deltax)
        self.varv = math.sqrt(3/deltav)
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
        self.weights = np.random.uniform(low=-0.3, high=0.3, size=(nx*nv,)).reshape((nx*nv,1))

    def getFeatureVector(self,state):
        input_features = []
        for i in range(len(self.features)):
            input_features.append(self.features[i].calc(state))
        input_features = np.asarray(input_features).reshape((len(self.features), 1))
        return input_features

    def feed(self, state):
        input = self.getFeatureVector(state)
        res = self.weights.transpose().dot(input)
        return res

if __name__ == '__main__':
    learning_rate = 1.5
    
    g = 9.82
    m = 1
    h = 1
    T = 0.05
    am = 4
    ksi = 1
    tf = 10
    nx = 5
    nv = 5
    L = 4
    vm = math.sqrt(2*g*h)

    Q = {-1:RBF(nx,nv,L,vm), 1:RBF(nx,nv,L,vm)}


    NumberOfEpisodes = 20   
    episode = 0
    epsilon0 = 0.9
    curTime = 0
    reward = 0
    simulator = sim(0.05)
    plot = Plotter(L,h)

    
    succesCounter = 0
    succesrate = 0

    while episode<NumberOfEpisodes:
        episode+=1
        epsilon = epsilon0/episode
       
        if episode == 0:
            succesrate = 0
        else: 
            succesrate = (succesCounter/episode)*100
        
        print("Episode: ",episode, " Reward: ",reward, " Actions:",curTime/T, " Succesrate: ",succesrate, " %")   
        reward = 0
        curTime = 0
        st = [L/2,0]
        simulator.currentX = simulator.startX
        simulator.currentV = simulator.startVel
        at = int(getNextAction(Q, st, epsilon))
      
        while curTime<tf and st[0] < L and st[0]>0:
            # print("State: ",st)
            
            stnext, rnext = simulator.step(at)
            reward = rnext

            if stnext[0]>=L:
                print("Goal reached")
                succesCounter += 1
                W = Q[at].weights
                W = W + learning_rate*(rnext-Q[at].feed(st))*Q[at].getFeatureVector(st)
                Q[at].weights = W
                break

            anext = int(getNextAction(Q, stnext, epsilon))
            W = Q[at].weights
            W = W + learning_rate * (rnext + ksi*Q[anext].feed(stnext) - Q[at].feed(st)) * Q[at].getFeatureVector(st)
            Q[at].weights = W
            st = stnext
            at = anext
            
            plot.plot(st[0])
         
            curTime += T

            
            # print(at)
            # print("Time",curTime)
            # print("Pos , vel",st)

    # print(W)