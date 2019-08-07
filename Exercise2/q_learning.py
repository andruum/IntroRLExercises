import operator
import random
from copy import deepcopy
from enum import Enum

import numpy as np

COLUMNS = 4
ROWS = 3

environment = [ [ ' ', ' ', ' ', '+' ],
                [ ' ', '#', ' ', '-' ],
                [ ' ', ' ', ' ', ' ' ] ]

V = [[0 for i in range(4)] for i in range(3)]

class State:
    def __init__(self, x=0, y=0, outside=False):
        self.x = x
        self.y = y
        self.is_outside_environment = outside
    def __hash__(self):
        return hash(str(self.x)+" "+str(self.y))
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

TERMINAL_STATE = State(-1,-1,True)

discount_rate = 0.9
theta = 0.01

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def GetNextState(state, action):
    state = deepcopy(state)

    if action == Actions.UP:
        state.y -= 1
    elif action == Actions.DOWN:
        state.y += 1
    elif action == Actions.LEFT:
        state.x -= 1
    elif action == Actions.RIGHT:
        state.x += 1

    if state.x < 0 or state.y < 0 or state.x >= COLUMNS or state.y >= ROWS:
        return TERMINAL_STATE

    if environment[state.y][state.x] == '#':
        state.is_outside_environment = True
    else:
        state.is_outside_environment = False

    return state

def GetReward(state, action):
    next = GetNextState(state, action)
    if next.is_outside_environment:
        return -1.0
    else:
        if environment[next.y][next.x] == '+':
            return 1.0
        if environment[next.y][next.x] == '-':
            return -1.0
        return -0.1

def GetNextAction(state_action_values, epsilon):
    probabilites = deepcopy(state_action_values)
    max_action = max(state_action_values.items(), key=operator.itemgetter(1))[0]

    for action in state_action_values:
        if action == max_action:
            probabilites[action] = 1 - epsilon + epsilon/len(state_action_values)
        else:
            probabilites[action] = epsilon / len(state_action_values)

    p = []
    for prob in probabilites.values():
        p.append(float(prob))

    best_action_id = np.random.choice(len(probabilites.keys()), 1, p=p)[0]
    return list(state_action_values.keys())[best_action_id]

def PrintEnvironment():
    for y in range(-1,ROWS+1):
        for x in range(-1,COLUMNS+1):
            if (y < 0 or y >= ROWS or x < 0 or x >= COLUMNS):
                print("#", end='')
            else:
                print(environment[y][x], end='')
        print("")

def PrintStateValues():
    for y in range(0, ROWS):
        for x in range(0, COLUMNS):
            print(" ",V[y][x], " ", end='')
        print("")

def CheckTermination(state):
    if state == TERMINAL_STATE:
        return True
    if state.is_outside_environment:
        return True
    if environment[state.y][state.x] == '#':
        return True
    if environment[state.y][state.x] == '+':
        return True
    if environment[state.y][state.x] == '-':
        return True

if __name__ == '__main__':
    print("Environment:")
    PrintEnvironment()

    Q = {}
    for y in range(0, ROWS):
        for x in range(0, COLUMNS):
            state = State(x, y)
            Q[state] = {Actions.RIGHT:0, Actions.LEFT:0, Actions.DOWN:0, Actions.UP:0}

    epsilon = 0.1
    alpha = 0.9
    sigma = 0.5
    episodes = 30
    for ep in range(episodes):
        state = State(0,2)

        if ep == episodes - 1:
            print("Final episode")

        while not CheckTermination(state):
            action = GetNextAction(Q[state],epsilon)
            reward = GetReward(state,action)
            next = GetNextState(state,action)

            max_next_state_value = -1
            if not next.is_outside_environment:
                max_next_state = Q[next]
                max_next_state_value = max(max_next_state.items(), key=operator.itemgetter(1))[1]

            Q[state][action] = Q[state][action] + alpha*(reward+sigma*max_next_state_value-Q[state][action])

            if ep == episodes-1:
                print(state.x, state.y, action)

            state = next