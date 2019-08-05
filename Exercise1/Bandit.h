#pragma once

#include <random>

class Bandit {
public:
    virtual int pull()=0;
};

class ConstantBandit: public Bandit{
public:
    void init(int reward);
    int pull() override;
private:
    int reward_;
};

class RangeBandit: public Bandit{
public:
    void init(int lowerReward, int upperReward);
    int pull() override;
private:
    std::mt19937 gen_;
    std::uniform_int_distribution<int> dis_;
};