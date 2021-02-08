# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:54:04 2020

@author: youle
"""

from fractions import Fraction
import numpy as np

def hits(m, h):

    a_ = np.dot(m, h)
    a_ /= max(a_)

    h_ = np.dot(m.T, a_)
    h_ /= max(h_)
    return a_, h_


mat = np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]])
# mat = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0]])

h = np.ones(mat.shape[0])

for i in range(100):

    a, h = hits(mat, h)

# print(*list(map(lambda y: Fraction(y).limit_denominator(1000), x)))