# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:12:57 2020

@author: youle
"""

import numpy as np

from kp_single import kp_single_obj
from TDGA_ext import TDGA_ext

p = kp_single_obj()

alg = TDGA_ext(30, 50, 50, p, 20)

alg.init_pop()
data = alg.execute(5000)