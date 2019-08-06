import random
import time

class Bandit:

    def pull(self):
        pass

class ConstantBandit(Bandit):

    def __init__(self, reward):
        super(ConstantBandit, self).__init__()
        self.reward = reward

    def pull(self):
        return self.reward

class RangeBandit(Bandit):

    def __init__(self, lowerReward, upperReward):
        super(RangeBandit, self).__init__()
        self.lowerReward = lowerReward
        self.upperReward = upperReward
        random.seed(time.time())

    def pull(self):
        return random.randint(self.lowerReward, self.upperReward)


if __name__ == '__main__':
    bandits = []

    bandits.append(ConstantBandit(1))
    bandits.append(ConstantBandit(3))
    bandits.append(RangeBandit(3,5))
    bandits.append(RangeBandit(6,8))

    for bandit in bandits:
        print(bandit.pull())