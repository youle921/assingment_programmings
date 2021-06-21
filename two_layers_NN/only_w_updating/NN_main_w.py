# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:53:55 2021

@author: youle
"""
import numpy as np
import matplotlib.pyplot as plt

from two_NN_w import neural_network

ans, data = np.split(np.loadtxt("input_data.dat"), [1], axis = 1)

n_node = [data.shape[1], 20, ans.shape[1]]
itr = 10000

np.random.seed(0)
solver = neural_network(n_node, lr = 0.01)

loss = [*map(lambda l: l.mean(), solver.execute(data, ans, itr))]

c_names = ["lightsteelblue", "lightsalmon"]

div = 1000
mesh = np.linspace(0, 10, div)
grid = np.array([[i,j] for i in mesh for j in mesh])

out = solver.predict(grid)
out_label = out.argmax(axis = 1)

Y, X = np.meshgrid(mesh, mesh)

plt.figure()

plt.scatter(*grid.T, c = out, s = 0.1, marker = ',', vmin = 0, vmax = 1, cmap = plt.cm.get_cmap('RdYlBu_r'))
plt.colorbar()

for i in range(2):
    plt.scatter(*data[ans.squeeze() == i].T, s = 100, edgecolors = 'k', label = F'Input: class{i+1}')

plt.legend(bbox_to_anchor=(1.20, 1))
plt.gca().set_aspect("equal")

plt.xlim([0, 10])
plt.ylim([0, 10])

plt.show()
plt.savefig("result_fool.png", dpi = 720, bbox_inches = 'tight')