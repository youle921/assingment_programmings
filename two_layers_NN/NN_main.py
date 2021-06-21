# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:53:55 2021

@author: youle
"""
import numpy as np
import matplotlib.pyplot as plt

from two_NN import neural_network

ans, data = np.split(np.loadtxt("input_data.dat"), [1], axis = 1)
one_hot_ans = np.zeros([ans.shape[0], 2])
one_hot_ans[ans.squeeze() == 0, 0] = 1
one_hot_ans[ans.squeeze() == 1, 1] = 1

n_node = [data.shape[1], 10, one_hot_ans.shape[1]]
itr = 10000

np.random.seed(0)
solver = neural_network(n_node, lr = 0.1)

loss = [*map(lambda l: l.mean(), solver.execute(data, one_hot_ans, itr))]

c_names = ["lightsteelblue", "lightsalmon"]

div = 1000
mesh = np.linspace(0, 10, div)
grid = np.array([[i,j] for i in mesh for j in mesh])

out = solver.predict(grid)
out_label = out.argmax(axis = 1)

Y, X = np.meshgrid(mesh, mesh)

plt.figure()

for i in range(2):
    plt.scatter(*grid[out_label == i].T,s = 1, c = c_names[i], marker = ',', label = F"Predict: class{i+1}")
plt.contour(X, Y, out_label.reshape([div, div]), colors = 'k')
for i in range(2):
    plt.scatter(*data[ans.squeeze() == i].T, s = 100, edgecolors = 'k', label = F'Input: class{i+1}')

plt.legend(bbox_to_anchor=(1.05, 1))
plt.gca().set_aspect("equal")

plt.show()