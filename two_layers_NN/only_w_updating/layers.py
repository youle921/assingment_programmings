# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:12:09 2021

@author: youle
"""

import numpy as np
from abc_layer_class import abc_layer

class affine(abc_layer):

    def __init__(self, n_in, n_out):

        self.W = np.random.randn(n_in, n_out)
        self.b = np.zeros(n_out)

        #for backpropagation(parameters updating)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T , dout)
        self.db = dout.sum(axis = 0)

        return dx

    def update_params(self, lr):

        self.W -= lr * self.dW
        self.b -= lr * self.db

class affine_w(abc_layer):

    def __init__(self, n_in, n_out):

        self.W = np.random.randn(n_in, n_out)
        self.b = np.ones(n_out) * 0.5

        #for backpropagation(parameters updating)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T , dout)
        self.db = dout.sum(axis = 0)

        return dx

    def update_params(self, lr):

        self.W -= lr * self.dW
        # self.b -= lr * self.db

class sigmoid(abc_layer):

    def __init__(self):

        self.out = None

    def forward(self, x):

        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):

        dx = dout * (1 - self.out) * self.out

        return dx

    def update_params(self, lr):

        pass

class squared_error:

    def __init__(self):

        self.x = None
        self.ans = None

    def forward(self, x, t):

        out = (x - t)**2
        self.x = x

        return out

    def backward(self, t):

        return 2 * (self.x - t)