# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:34 2020

@author: youle
"""

import numpy as np

def bloom(k, n, m):
    bit = np.zeros(n)
    div = int(n / k)

    for x in range(m):
        for i in range(k):
            bit[np.random.randint(div) + i * div] = 1

    check = [[x] for x in range(div)]

    for i in range(k-1):
        check = [ x + [y] for x in check for y in range(div)]

    c = map(lambda x: bit[x[0]] * bit[x[1] + div] * bit[x[2] + 2*div], check)

    return sum(c)/len(check)

l = []
for i in range(100):
    l.append(bloom(3, 200, 100))
print(sum(l)/len(l))

