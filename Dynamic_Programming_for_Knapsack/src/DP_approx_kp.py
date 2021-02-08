# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 03:25:45 2021

@author: youle
"""
import numpy as np

cap_idx = 0
prof_idx = 1

def get_NDsolution(sol):

    # for minmization problem(compalator: >)

    sol_ = sol.copy()
    sol_[:, prof_idx] *= -1
    is_dominated = (sol_[:, None, :] >= sol_[None, :, :]).prod(axis = 2) & \
        (sol_[:, None, :] > sol_[None, :, :]).max(axis = 2)

    NDsolution = is_dominated.max(axis = 1) == 0

    return sol[NDsolution]

def DP_knapsack(items, capacity):

    knapsack = np.zeros([1, 2])
    knapsack = np.vstack((knapsack, items[0]))
    n = items.shape[0]

    for i in range(1, n):

        candidate = knapsack + items[i]
        feasible = candidate[:, cap_idx] <= capacity
        if feasible.sum() > 0:
            knapsack = get_NDsolution(np.vstack([knapsack, candidate[feasible]]))

    return knapsack[:, prof_idx].max()

def DP_knapsack_approx(items, capacity, epsilon):

    items_ = items.copy()
    mu = (epsilon * items_[:, prof_idx].max()) / items_.shape[0]
    items_[:, prof_idx] = items_[:, prof_idx] / mu

    return DP_knapsack(items_, capacity) * mu

np.random.seed(2021)
items = np.random.randint(1, 11, size = [10, 2])
capacity = 25
eps = 0.5
max_profit = DP_knapsack_approx(items, capacity, eps)

print("max profit is {0}".format(max_profit))

