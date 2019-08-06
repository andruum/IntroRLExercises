from copy import deepcopy
from enum import Enum

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
    if (environment[state.y][state.x] != ' '):
        return TERMINAL_STATE
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
    state.is_outside_environment = False
    return state

def GetReward(state, action):
    next = GetNextState(state, action)
    if next.is_outside_environment:
        return 0
    else:
        if environment[next.y][next.x] == '+':
            return 1.0
        if environment[next.y][next.x] == '-':
            return -1.0
        return 0

def GetNextAction(state):
    return Actions.RIGHT


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

if __name__ == '__main__':
    print("Environment:")
    PrintEnvironment()

    sweep = 0
    delta = 0

    while True:
        for y in range(0, ROWS):
            for x in range(0, COLUMNS):
                state  = State(x,y)
                if environment[y][x] == ' ':
                    v = V[y][x]
                    action = GetNextAction(state)
                    reward = GetReward(state, action)
                    next   = GetNextState(state, action)
                    if not next.is_outside_environment:
                        V[y][x] = reward + discount_rate * V[next.y][next.x]
                    delta = max(delta, abs(v - V[y][x]))
        sweep += 1
        print("Sweep #", sweep, "delta", delta)
        PrintStateValues()
        if delta > theta:
            break