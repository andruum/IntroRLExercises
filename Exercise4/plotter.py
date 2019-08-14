import time

import math
import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, L=4, h=1):
        self.h = h
        self.L = L
        plt.ion()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        x = np.linspace(0, L, 100)
        y = h / 2 * (1 + np.cos(2 * math.pi * x / L))
        mountains, = ax.plot(x, y, 'b-')
        self.car_point, = ax.plot(0, 0, 'r.', markersize=12)

    def plot(self,x):
        self.car_point.set_xdata(x)
        self.car_point.set_ydata(self.y(x))
        self.fig.canvas.draw()

    def y(self,x):
        return self.h/2 * (1+np.cos(2*math.pi*x/self.L))


if __name__ == '__main__':
    plot = Plotter(4,1)
    for i in np.linspace(0, 4, 100):
        plot.plot(i)