# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:54:52 2020

@author: youle
"""

import numpy as np
import itertools

def get_basket():

    basket = [[1] for i in range(100)]

    for b in range(1, 100):
        for i in range(2,  b + 2):
            if (b + 1) % i == 0:
                basket[b].append(i)

    return basket

def get_basket_2():

    basket = [[] for i in range(100)]

    for b in range(100):
        for i in range(1,  int(100/(b+1)) + 1):
            basket[b].append(i * b + i)

    return basket

def apriori(arr):

    num = 1
    freq = []

    out = []

    for i in range(1, 101):
        if sum(map(lambda x: i in x, b)) > 4:
            freq.append([i])

    while (freq != []):

        freq.sort()
        out.append(freq)

        freq_elem = set(sum(freq, []))
        freq = []
        num += 1
        comb = itertools.combinations(freq_elem, num)

        for i in comb:
            if sum( map(lambda x: set(i).issubset(set(x)), arr) ) > 4:
                freq.append([*i])

    return out

b = get_basket()

o = apriori(b)



