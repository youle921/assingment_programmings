# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:57:18 2020

@author: youle
"""
import numpy as np
import pandas as pd

from functools import partial

import matplotlib.pyplot as plt

def plot_basis(spl_mdl, lower = 0, upper = 10):

    num = 10000
    x = np.arange(num + 1)*(10/num)
    outs = map(lambda f: f(x), spl_mdl.basis)

    label = [*map (lambda i: "h" + str(i) + "(x)", range(len(spl_mdl.basis)))]

    [*map(lambda y, l: plt.plot(x, y, label = l), outs, label)]
    plt.legend()

    return

def load_data():

    tmp = pd.read_csv("train.csv")
    x = tmp["x"].to_numpy()
    y = tmp["y"].to_numpy()

    return x, y

class cubic_spline:

    def __init__(self, k, lower = 0, upper = 10):

        knot = np.linspace(lower, upper, num = k + 2)[1:k + 1]

        self.basis = [*map (lambda exp: (lambda x: x**exp), range(4))]
        self.basis += [*map(lambda c: (lambda x: np.maximum(x - c, 0)**3), knot)]
        self.calc_x = lambda x: np.array([*map(lambda f: f(x).tolist(), self.basis)]).T

    def calc_beta(self, data, ans):

        x = self.calc_x(data)
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), ans)

    def predict(self, data):

        x = self.calc_x(data)
        out = np.dot(x, self.beta)

        return out

    def calc_magic_fomula(self, data, ans):

        x = self.calc_x(data)
        h_mat = np.dot(np.dot(x, np.linalg.inv(np.dot(x.T, x))), x.T)
        out = self.predict(data)
        loo = (((ans - out)/ (1 - np.diag(h_mat)))**2).mean()

        return loo

def calc_loss(spl_mdl, data, ans, idx):

    mask = np.arange(data.shape[0]) != idx
    spl.calc_beta(data[mask], ans[mask])

    return (ans[idx] - spl.predict(data[idx]))**2

def calc_magic_loo(data, ans, k):

    spl = cubic_spline(k)
    spl.calc_beta(data, ans)
    return spl.calc_magic_fomula(data, ans)

data, ans = load_data()

spl = cubic_spline(4)
spl.calc_beta(data, ans)
out = spl.predict(np.arange(0,101)/10)

magic_loo = spl.calc_magic_fomula(data, ans)
loo = np.mean([*map(partial(calc_loss, spl, data, ans),\
                    range(data.shape[0]))])

k = range(1, 16)
magic_loo_list = [*map(partial(calc_magic_loo, data, ans), k)]
best_k = k[np.argmin(magic_loo_list)]