import math

from Bandit import *
<<<<<<< Updated upstream:Algorithm.py
import numpy
=======
import numpy as np
>>>>>>> Stashed changes:Exercise1/Algorithm.py
import random
import operator


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
        return numpy.where(Q == numpy.amax(Q))[0][0]

class GreedyAlgorithm(Algorithm):

    def __init__(self, e, totalnum, k):
        self.Q = [0 for i in range(k)]
        self.N = [0 for i in range(k)]
        self.e = e
        self.totalnum = totalnum

    def generateDecision(self, totalbandits):
        rand = random.uniform(0, 1)
        if rand < self.e :
            bandit = random.randint(0, totalbandits-1)
            return bandit
        else:
            return numpy.where(self.Q == numpy.amax(self.Q))[0][0]

    def evaluate(self, bandits):
        for i in range(self.totalnum):
            A = self.generateDecision(len(bandits))
            R = bandits[A].pull()
            self.N[A] += 1
            self.Q[A] += (R-self.Q[A])/self.N[A]
        return numpy.where(self.Q == numpy.amax(self.Q))[0][0]

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
            tempQ.append(self.Q[i] + self.c*math.sqrt(t/self.N[i]))
        return numpy.where(tempQ == numpy.amax(tempQ))[0][0]


if __name__ == '__main__':
    bandits = []

    bandits.append(ConstantBandit(1))
    bandits.append(ConstantBandit(3))
    bandits.append(RangeBandit(3, 5))
    bandits.append(RangeBandit(3, 6))

    algo = UpperConfidenceAlgorithm(0, 1000, len(bandits), 2)

<<<<<<< Updated upstream:Algorithm.py
    print(algo.evaluate(bandits)+1)
    print("dab")
=======
    plt.legend()
    plt.show()
    plt.savefig('plot2-better.png')
>>>>>>> Stashed changes:Exercise1/Algorithm.py
