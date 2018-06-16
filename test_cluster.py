#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 00:42:53 2018

@author: elvex
"""

import numpy as np
import numpy.random as npr
import random
 
def init_board(N, mini = -1, maxi = 1):
    X = npr.uniform(mini, maxi (N, 2))
    return X

def init_board_gauss(N, k, mini = -1, maxi = 1, ecart_min = 0.05, ecart_max = 0.10):
    n = N//k
    X = []
    for i in range(k):
        centre, s = npr.uniform(mini, maxi, 2), random.uniform(ecart_min, ecart_max)
        x = npr.normal(centre, s, (n, 2))
        X.append(x)
    X = np.vstack(X)
    return X