# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:17:08 2021

@author: t.urita
"""
import numpy as np

from kp import knapsack

n_items = 100
kp_obj = knapsack(items = n_items)

n_samples = 10000

variables = np.random.randint(0, 2, size = [n_samples, n_items])
_, vio = kp_obj.evaluate_with_violation(variables)

ans = np.zeros(n_samples)

c3 = vio.sum(axis = 1) == 0
c2 = (vio[:, 0] == 0) & (vio[:, 1] != 0)
c1 = (vio[:, 0] != 0) & (vio[:, 1] == 0)

ans[c3] = 3
ans[c2] = 2
ans[c1] = 1

np.savetxt("train.csv", variables, delimiter = ",")
np.savetxt("answer.csv", ans, delimiter = ",")