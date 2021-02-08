# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:39:39 2020

@author: youle
"""

import numpy as np

from kp_single import kp_single_obj
from SGA import SGA

p = kp_single_obj()

alg = SGA(30, 50, 50, p)

alg.init_pop()
data = alg.execute(5000)