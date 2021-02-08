# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:57:51 2020

@author: youle
"""

import numpy as np

class kp_single_obj:

    def __init__(self, min_flag = True):

        self.items = np.array([
                                (75, 7),
                                (84, 9),
                                (58, 13),
                                (21, 5),
                                (55, 16),
                                (95, 28),
                                (28, 15),
                                (76, 43),
                                (88, 60),
                                (53, 37),
                                (58, 44),
                                (81, 63),
                                (32, 34),
                                (89, 95),
                                (54, 61),
                                (23, 29),
                                (42, 57),
                                (52, 72),
                                (58, 83),
                                (53, 84),
                                (30, 48),
                                (26, 45),
                                (40, 74),
                                (40, 78),
                                (26, 52),
                                (39, 79),
                                (25, 64),
                                (23, 64),
                                (16, 55),
                                (12, 74)
                            ])

        self.capacity = 744
        self.sign = 1 - 2*min_flag

        self.opt_sol = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
        self.opt_f = np.dot(self.items[:, 0], self.opt_sol)

    def evaluate(self, solution):

        f, w = *(np.dot(solution, self.items).T),
        f[w > self.capacity] = 0

        return f * self.sign

    def analyze(self, solution):

        f, w = *(np.dot(solution, self.items).T),

        max_f = np.max(f)
        average_f = np.mean(f[w < self.capacity])
        opt_num = np.sum(f == self.opt_f)

        return max_f, average_f, opt_num

