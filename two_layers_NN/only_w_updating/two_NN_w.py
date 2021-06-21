# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:25:03 2021

@author: youle
"""
import numpy as np
from collections import OrderedDict

import layers

class neural_network:

    def __init__(self, n_node, lr = 0.01):

        #store larning rate
        self.lr = lr

        #initialize layers
        self.layers = OrderedDict()

        for i in range(1, len(n_node)):

            self.layers[F'affine{i}'] = layers.affine_w(n_node[i - 1], n_node[i])
            self.layers[F'sigmoid{i}'] = layers.sigmoid()

        self.loss_function = layers.squared_error()

    def predict(self, data):

        out = data
        for layer in self.layers.values():
            out = layer.forward(out)

        return out

    def execute(self, data, ans, n_iter):

        loss = [self.forward(data, ans)]

        for itr in range(n_iter):

            for i in range(data.shape[0]):

                self.forward(data[i:i + 1], ans[i: i + 1])
                self.backward(ans[i:i + 1])
                self.learning()

                loss.append(self.forward(data, ans))

        return loss

    def forward(self, data, ans):

        out = self.predict(data)

        return self.loss_function.forward(out, ans)

    def backward(self, a):

        d_out = self.loss_function.backward(a)

        r_layers = list(self.layers.values())
        r_layers.reverse()

        for layer in r_layers:
            # if d_out.size == 1:
            #     d_out = d_out[0]
            d_out = layer.backward(d_out)

    def learning(self):

        [*map(lambda layer: layer.update_params(self.lr), self.layers.values())]