# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:16:09 2021

@author: t.urita
"""
import numpy as np
import os

class knapsack:

    def __init__(self, objective = 2, items = 500, lower = 10, upper = 100, load = True):

        self.objective = objective
        self.pf = None

        if load:
            path = os.path.dirname(__file__)
            self.items = np.empty([objective, items, 2])
            for i in range(objective):
                load_item = np.loadtxt(path + "/items/knapsack_500_profit" + str(i + 1) + ".csv")
                self.items[i, :, 0] = load_item[:items]
                load_weight = np.loadtxt(path + "/items/knapsack_500_weight" + str(i + 1) + ".csv")
                self.items[i, :, 1] = load_weight[:items]

        else:
            self.items = np.random.randint(lower, high = upper + 1, size = (items, 2, objective))

        self.utility = (self.items[0:2, :, 0] / self.items[0:2, :, 1]).max(axis = 0)

        self.size = np.sum(self.items[:, :, 1], axis = 1) / 2
        if objective > 2:
            self.size[2:] *= 2

        self.original_size = self.size
        self.repair_order = (self.items[0:2, :, 0] / self.items[0:2, :, 1])

    def evaluate_with_violation(self, solutions):

        f = np.dot(solutions, self.items[:, :, 0].T)
        w = np.dot(solutions, self.items[:, :, 1].T)

        return [-f, np.clip(w - self.size, 0, None)]