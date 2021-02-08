# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:02:48 2020

@author: youle
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

class k_center_problem:

    def __init__(self, n, dim = 2):

        self.point = np.random.rand(n, 2)

    def calc_square_dist(self, k):

        return ((self.point[None, :, :] - k[:, None, :])**2).sum(axis = 2)

    def calc_distance_from_center(self, k):

        sq_dist = self.calc_square_dist(k)

        return sq_dist.min(axis = 0)**0.5

    def assign_center(self, k):

        sq_dist = self.calc_square_dist(k)

        return sq_dist.argmin(axis = 0)

    def get_random_point(self):

        return self.point[np.random.choice(self.point.shape[0], 1)]

class greedy_solver:

    def __init__(self, k, dim = 2):

        self.center = np.empty([k, dim])

    def solve(self, prob):

        self.center[0] = prob.get_random_point()
        for i in range(1, self.center.shape[0]):
            dist = prob.calc_square_dist(self.center[:i])
            max_idx = dist.min(axis = 0).argmax()
            self.center[i] = prob.point[max_idx]

        c = prob.assign_center(self.center)

        r = np.zeros(self.center.shape[0])
        dist = prob.calc_square_dist(self.center)
        for i in range(self.center.shape[0]):
            r[i] = dist[i][c == i].max()

        return [self.center, r**0.5]

np.random.seed(1)
prob = k_center_problem(15)
solver = greedy_solver(int(15**0.5)+1)
center, r= solver.solve(prob)

assigned_center = prob.assign_center(center)

fig = plt.figure()
ax = plt.axes()
ax.scatter(*prob.point.T, c = "k")
for i in range(int(15**0.5)+1):
    ax.scatter(*center[i], label = "Center" + str(i+1))
    circle = ptc.Circle(xy = center[i], radius = r[i], fill = False)
    ax.add_patch(circle)

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal')
ax.legend()