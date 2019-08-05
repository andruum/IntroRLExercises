#include "Bandit.h"
#include <time.h>
#include <cstdlib>
#include <random>

void ConstantBandit::init(int reward) {
    this->reward_=reward
}

int ConstantBandit::pull() {
    return this->reward_;
}

void RangeBandit::init(int lowerReward, int upperReward) {
    std::random_device rd;
    this->gen_ = std::mt19937(rd());
    this->dis_ = std::uniform_int_distribution<int>(lowerReward,upperReward);
}

int RangeBandit::pull() {
    return this->dis_(this->gen_);
}