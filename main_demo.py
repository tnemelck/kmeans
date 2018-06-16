#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:23:56 2018

@author: elvex
"""

from kmeans import Kmeans
import test_cluster as tc

def main():
    img_dir = '/home/elvex/utt/SRT6/IF25/projet/img_test_1000_4'
    km_path = '/home/elvex/utt/SRT6/IF25/projet/KM/test_1000_4'
    
    km = Kmeans(tc.init_board_gauss(1000, 4, -100, 100, 5, 10), nb_cluster=20, cpu=8, methode_dist="euclidean", adr=img_dir)
    km.run_global(choose_nb_graph=True, grphq=True)
    
    km.save(km_path)
    
    return km