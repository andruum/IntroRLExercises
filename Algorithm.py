from Bandit import *
import numpy

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



if __name__ == '__main__':
    bandits = []

    bandits.append(ConstantBandit(1))
    bandits.append(ConstantBandit(3))
    bandits.append(RangeBandit(3, 5))
    bandits.append(RangeBandit(3, 5))

    algo = UniformAlgorithm(50)

    print(algo.evaluate(bandits))