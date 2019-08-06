import math

from Exercise1.Bandit import *
import numpy as np
import random
import matplotlib.pyplot as plt

class Algorithm:

    def evaluate(self, bandits):
        pass


class UniformAlgorithm(Algorithm):

    def __init__(self, w):
        self.w = w

    def evaluate(self, bandits):
        Q = []
        for bandit in bandits:
            Qcur = 0
            for i in range(self.w):
                reward = bandit.pull()
                Qcur += reward
            Q.append(Qcur)
        return np.where(Q == np.amax(Q))[0][0]

class GreedyAlgorithm(Algorithm):

    def __init__(self, e, totalnum, k):
        self.Q = [0 for i in range(k)]
        self.N = [0 for i in range(k)]
        self.data = []
        self.e = e
        self.temp = 0
        self.totalnum = totalnum

    def generateDecision(self, totalbandits):
        rand = random.uniform(0, 1)
        if rand < self.e :
            bandit = random.randint(0, totalbandits-1)
            return bandit
        else:
            return np.where(self.Q == np.amax(self.Q))[0][0]


    def evaluate(self, bandits):
        for i in range(self.totalnum):
            A = self.generateDecision(len(bandits))
            R = bandits[A].pull()
            self.N[A] += 1
            self.Q[A] += (R-self.Q[A])/self.N[A]
            self.temp += R
            self.data.append(self.temp/sum(self.N))
        return np.where(self.Q == np.amax(self.Q))[0][0]

class OptimisticGreedyAlgorithm(GreedyAlgorithm):

    def __init__(self, e, totalnum, k, Q0):
        super(OptimisticGreedyAlgorithm, self).__init__(e,totalnum,k)
        self.Q = [Q0 for i in range(k)]

class UpperConfidenceAlgorithm(GreedyAlgorithm):

    def __init__(self, e, totalnum, k, c):
        super(UpperConfidenceAlgorithm, self).__init__(e,totalnum,k)
        self.c = c

    def generateDecision(self, totalbandits):
        t = sum(self.N)+1
        tempQ = []
        for i in range(len(self.Q)):
            if self.N[i] == 0:
                return i
            tempQ.append(self.Q[i] + self.c*(math.sqrt(t/self.N[i])))
        return np.where(tempQ == np.amax(tempQ))[0][0]


if __name__ == '__main__':
    bandits = []

    bandits.append(ConstantBandit(1))
    bandits.append(RangeBandit(4, 6))
    bandits.append(ConstantBandit(3))
    bandits.append(RangeBandit(-10, 10))
    bandits.append(RangeBandit(-3, 6))
    bandits.append(ConstantBandit(-1))
    bandits.append(ConstantBandit(0))
    bandits.append(RangeBandit(-1,1))
    bandits.append(RangeBandit(-20,20))
    bandits.append(RangeBandit(-2,3))

    totalsteps = 1000

    algo1 = GreedyAlgorithm(0.1, totalsteps, len(bandits))
    algo2 = OptimisticGreedyAlgorithm(0, totalsteps, len(bandits), 20)
    algo3 = UpperConfidenceAlgorithm(0, totalsteps, len(bandits), 0.2)

    best1 = algo1.evaluate(bandits)
    best2 = algo2.evaluate(bandits)
    best3 = algo3.evaluate(bandits)

    print(best1,best2,best3)

    plt.plot(range(totalsteps),np.asarray(algo1.data), label="GreedyAlgorithm")
    plt.plot(range(totalsteps),np.asarray(algo2.data), label="OptimisticGreedyAlgorithm")
    plt.plot(range(totalsteps),np.asarray(algo3.data), label="UpperConfidenceAlgorithm")

    plt.xlabel('Step')
    plt.ylabel('Awerage reward')

    plt.legend()
    plt.show()