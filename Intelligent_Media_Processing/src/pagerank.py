# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:06:46 2020

@author: youle
"""

from fractions import Fraction
import numpy as np

def pagerank(m, x, beta, trust):

    mat = m*beta

    const = np.zeros_like(x)
    const[trust] = 1
    const = const/sum(const) *(1 - beta)

    return np.dot(mat.T, x) + const


# mat = np.array([[1/3, 1/3, 1/3], [1/2, 0, 1/2], [0, 1/2, 1/2]])
mat = np.array([[0, 1/3, 1/3, 1/3], [1/2, 0, 0, 1/2], [1, 0, 0, 0], [0, 1/2, 1/2, 0]])

x = np.ones(mat.shape[0])/mat.shape[0]
trust = [0,1,2,3]

for i in range(100):
    # x_ = x
    x = pagerank(mat, x, 0.8, trust)
    # print(x)
page = list(map(lambda y: Fraction(y).limit_denominator(1000), x))

trust = [1]
for i in range(100):
    # x_ = x
    x = pagerank(mat, x, 0.8, trust)
    # print(x)

trust = list(map(lambda y: Fraction(y).limit_denominator(1000), x))

