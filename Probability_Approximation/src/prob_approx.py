# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:35:12 2020

@author: youle
"""
import numpy as np

np.random.seed(0)

sampling_num = 10000
sample = np.random.randn(sampling_num)

prob = map(lambda x: np.mean(sample[:x] > 2.5), \
           range(1, sampling_num + 1))

imp_sample = np.random.randn(sampling_num) + 3

f = lambda x: (np.exp(-x**2/2)) / np.sqrt(2*np.pi)
g = lambda x: (np.exp(-(x-3)**2/2)) / np.sqrt(2*np.pi)

imp_prob = map(lambda x: np.mean((imp_sample[:x] > 2.5) \
                                 * (f(imp_sample[:x])/g(imp_sample[:x]))), \
           range(1, sampling_num + 1))

import matplotlib.pyplot as plt

plt.plot(list(prob), label = "Monte Carlo")
plt.plot(list(imp_prob), label = "Importance Sampling")
plt.axhline(0.0062, c = "k", ls = "dashed", zorder = -1, label = "True Probability")

plt.legend()

