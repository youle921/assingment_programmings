# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:48:17 2020

@author: youle
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def two_dim_gaussian(x, mu, sigma):

    out = np.empty([mu.shape[1], x.shape[0]])
    for i in range(mu.shape[1]):

        det = np.linalg.det(sigma[i])
        inv = np.linalg.inv(sigma[i])

        mat = np.exp(- 0.5 * np.dot(np.dot((x - mu[i]), inv), (x - mu[i]).T)) / (2 * np.pi * det ** 0.5)
        out[i] = np.diag(mat)

    return out

class EM_gaussian:

    def __init__(self, dim, n_class, max_itr = 100):

        self.pi = np.ones(n_class) / n_class
        self.mu = np.random.randn(n_class, dim)
        self.sigma = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

        # self.pi = np.ones(n_class) / n_class
        # self.mu = np.array([[-1.5, 1],[1.5, -1]])
        # self.sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype = "float")

        self.n_class = n_class
        self.max_itr = max_itr
        self.eps = 1e-5

    def predict(self, x):

        prob = self.calc_probability(x)

        return np.argmax(prob, axis = 0)

    def fit(self, x):

        for t in range(self.max_itr):

            old_mu = self.mu.copy()
            pr = self.E_step(x)
            self.M_step(x, pr)

            if np.abs(old_mu - self.mu).mean() < self.eps:
                return

    def calc_probability(self, x):

        prob = self.pi[:, None] * two_dim_gaussian(x, self.mu, self.sigma)

        return prob / prob.sum(axis = 0)

    def E_step(self, x):

        return self.calc_probability(x)

    def M_step(self, x, pr):

        nk = pr.sum(axis = 1)
        self.mu = (x[None, :, :]*pr[:, :, None]).sum(axis = 1)/ nk

        for i in range(self.n_class):

            square = (x - self.mu[i])[:, :, None] * (x - self.mu[i])[:, None, :]
            prod = pr[i][:, None, None]
            self.sigma[i] = (square * prod).sum(axis = 0) / nk[i]

        self.pi = pr.mean(axis = 1)

x = pd.read_csv("ScaledOFData.csv").to_numpy()[:, 1:]
np.random.seed(0)
solver = EM_gaussian(2, 2)
solver.fit(x)

pred = solver.predict(x)
plt.figure(figsize = [3, 3])
plt.scatter(*x[pred == 0].T, label = "Class 1")
plt.scatter(*x[pred == 1].T, label = "Class 2")
# plt.scatter(*solver.mu[0], s = 200, marker = "*")
# plt.scatter(*solver.mu[1], s = 200, marker = "*")