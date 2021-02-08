# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:39:20 2021

@author: youle
"""
import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt

class cities:

    def __init__(self, num, seed):

        self.num_city = num
        self.city_pos = self.create_cities(seed)
        self.dist_matrix = distance.cdist(self.city_pos, self.city_pos, metric = 'euclidean')

    def create_cities(self, seed):

        np.random.seed(seed)

        return np.random.rand(self.num_city, 2)

    def plot_cities(self):

        fig, ax = plt.subplots()

        ax.scatter(*self.city_pos.T, label = ": cities")

        ax.set_aspect("equal")
        fig.legend()

        fig.show()

    def plot_pass(self, order):

        pos = np.array([*order, order[0]])

        fig, ax = plt.subplots()

        ax.scatter(*self.city_pos.T, label = ": cities")
        ax.plot(*self.city_pos[pos].T, c = "k", zorder = -1, label = ": pass")

        ax.set_aspect("equal")
        fig.legend()

        fig.show()

def nearest_addition(city_info):

    n = city_info.num_city
    dist = city_info.dist_matrix + np.eye(n)*2
    is_visited = [*np.unravel_index(np.argmin(dist), dist.shape)]
    not_visited = [i for i in range(n) if i not in is_visited]
    for i in range(n - 2):
        from_, next_ = np.unravel_index(np.argmin(dist[is_visited][:, not_visited]),\
                                       [len(is_visited), len(not_visited)])
        is_visited.insert(from_ + 1, not_visited.pop(next_))

    return is_visited

obj_cities = cities(10, 2021)
obj_cities.plot_cities()
order = nearest_addition(obj_cities)
obj_cities.plot_pass(order)
