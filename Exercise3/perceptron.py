import random
from copy import deepcopy

import math
import numpy as np

def linear(z):
    return z

def tlu(z):
    if z > 0:
        return 1
    else:
        return 0

def sigmoid(z):
    return 1/(1+math.exp(-z))

def hyper_tangent(z):
    return math.exp(z) - math.exp(-z) / (math.exp(z) + math.exp(-z))

class Perceptron:

    def __init__(self, num_inputs, activation_func):
        self.w = []
        for i in range(num_inputs):
            self.w.append(random.uniform(-1,1))
        self.w.append(random.uniform(-1,1))
        self.activation = activation_func
        self.output = []

    def feed(self, inputs):
        inputs = deepcopy(inputs)
        inputs.append(-1)
        z = 0
        for i in range(len(inputs)):
            z += self.w[i] * inputs[i]
        self.output = self.activation(z)
        return self.output


class MLP:

    def __init__(self, hidden_layer = [], output_layer=[]):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.input = []
        self.output = []

    def feed(self, input):
        self.input = input
        hidden_res = []
        for hidp in self.hidden_layer:
            hidden_res.append(hidp.feed(input))
        self.output = []
        for outp in self.output_layer:
            self.output.append(outp.feed(hidden_res))
        return self.output

    def back_propagate(self, targets, learning_rate):
        errors_k = []
        for k in range(len(self.output_layer)):
            t = targets[k]
            v = self.output[k]
            e = v*(1-v)*(t-v)
            errors_k.append(e)

        errors_j = []
        for j in range(len(self.hidden_layer)):
            hidn = self.hidden_layer[j]

            e = hidn.output*(1-hidn.output)
            sumeyk = 0
            for k in range(len(self.output_layer)):
                sumeyk += errors_k[k]*self.output_layer[k].w[j]
            e *= sumeyk
            errors_j.append(e)

        for k in range(len(self.output_layer)):
            neuron = self.output_layer[k]
            for j in range(len(neuron.w)-1):
                neuron.w[j] = neuron.w[j] + learning_rate*errors_k[k]*self.hidden_layer[j].output
            neuron.w[-1] = neuron.w[-1] + learning_rate*errors_k[k]*(-1)

        for j in range(len(self.hidden_layer)):
            neuron = self.hidden_layer[j]
            for i in range(len(neuron.w)-1):
                neuron.w[i] = neuron.w[i] + learning_rate*errors_j[i]*self.input[i]
            neuron.w[-1] = neuron.w[-1] + learning_rate*errors_j[j]*(-1)

        error_total = 0
        for ek in errors_k:
            error_total+= abs(ek)
        for ej in errors_j:
            error_total+= abs(ej)
        return error_total



u = [[0,0], [0,1], [1,0], [1,1]]
t = [[0],[1],[1],[0]]


if __name__ == '__main__':
    hidden_perc_1 = Perceptron(2,sigmoid)
    hidden_perc_2 = Perceptron(2,sigmoid)
    output_perc = Perceptron(2,sigmoid)

    mlp = MLP(hidden_layer=[hidden_perc_1,hidden_perc_2], output_layer=[output_perc])

    learning_rate = 0.99
    error = 1.0
    while error > 0.001:
        error = 0.0
        ids = np.arange(4)
        np.random.shuffle(ids)
        for i in ids:
            v = mlp.feed(u[i])
            error += mlp.back_propagate(t[i],learning_rate)
        # print(error)

    for i in range(4):
        v = mlp.feed(u[i])
        print(v)
        #todo draw decision boundary


if __name__ == '__main1__':
    perc = Perceptron(2,sigmoid)

    learning_rate = 0.1
    error = 1.0
    while error > 0.001:
        error = 0.0
        for i in range(4):
            v = perc.feed(u[i])

            for j in range(len(perc.w) - 1):
                perc.w[j] = perc.w[j] + learning_rate * v*(1-v)* (t[i][0]-v)*u[i][j]
            perc.w[-1] = perc.w[-1] + learning_rate * v*(1-v)* (t[i][0]-v) * (-1)

            error += 0.5*(t[i][0]-v)**2

    for i in range(4):
        v = perc.feed(u[i])
        print(v)
        #todo draw decision boundary