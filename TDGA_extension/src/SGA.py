# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:37:18 2020

@author: youle
"""

import numpy as np
from scipy.stats import norm

import numba

def uniform_crossover(parents):

    offspring1 = parents[0].copy()
    offspring2 = parents[1].copy()
    mask = (np.random.rand(offspring1.shape[0], 1) < 0.9) * (np.random.rand(*offspring1.shape,) < 0.5)
    offspring1[mask] = parents[1][mask]
    offspring2[mask] = parents[0][mask]

    return np.vstack((offspring1, offspring2))

def bitflip_mutation(offs):

    off = offs
    mutation_ratio = 1 / off.shape[1]
    mutation_mask = np.random.rand(*off.shape,) < mutation_ratio
    off[mutation_mask] = 1 - off[mutation_mask]

    return off

class SGA:

    def __init__(self, ndim, npop, noff, problem):

        self.pop = {}
        self.pop["variables"] = np.zeros([npop, ndim])
        self.pop["objectives"] = np.zeros(npop)

        self.problem = problem

        self.neval = 0
        self.noff = noff

        self.crossover = uniform_crossover
        self.mutation = bitflip_mutation

    def init_pop(self):

        self.pop["variables"] = np.random.randint(2, size = self.pop["variables"].shape)
        self.pop["objectives"] = self.problem.evaluate(self.pop["variables"])

        self.neval = self.pop["variables"].shape[0]

    @numba.jit
    def execute(self, max_eval):

        offs = {}
        offs["variables"] = np.zeros([self.noff, self.pop["variables"].shape[1]])
        offs["objectives"] = np.zeros(self.noff)

        n, mod= divmod(max_eval - self.neval, self.noff)

        analyze_data = np.empty([n, 3])

        for i in range(n):

            parents = []
            parents.append(self.pop["variables"][self.selection()])
            parents.append(self.pop["variables"][self.selection()])

            offs["variables"] = self.mutation(self.crossover(parents))
            offs["objectives"] = self.problem.evaluate(offs["variables"])
            self.update(offs)

            analyze_data[i] = self.problem.analyze(self.pop["variables"])

        if mod != 0:

            parents = []
            parents.append(self.pop["variables"][self.selection()])
            parents.append(self.pop["variables"][self.selection()])

            offs["variables"] = self.mutation(self.crossover(parents))
            offs["objectives"][:mod] = self.problem.evaluate(offs["variables"][:mod])
            offs["objectives"][mod:] = 0
            self.update(offs)

        self.neval = max_eval

        return analyze_data

    def selection(self):

        rand_idx = np.random.randint(self.pop["variables"].shape[0], size = [int(self.noff/2), 2])
        idx = np.argmin([self.pop["objectives"][rand_idx[:, 0]], self.pop["objectives"][rand_idx[:, 1]]], axis = 0)

        return rand_idx[range(rand_idx.shape[0]), idx]

    def update(self, offs):

        pop_size = self.pop["objectives"].shape

        union = {}
        union["variables"], idx = np.unique(np.vstack((self.pop["variables"] ,offs["variables"])),\
                                              axis = 0, return_index = True)
        union["objectives"] = np.hstack((self.pop["objectives"], offs["objectives"]))[idx]

        idx = np.argsort(union["objectives"])[:pop_size[0]]

        self.pop["variables"] = union["variables"][idx]
        self.pop["objectives"] = union["objectives"][idx]