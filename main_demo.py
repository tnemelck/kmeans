#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:23:56 2018

@author: elvex
"""

from kmeans import Kmeans
import test_cluster as tc

img_dir = './IMG/img_test_100_4'
km_path = './KM/test_100_4'
nb_points = 100
nb_classe = 4
mini = -100 
maxi = 100 
ecart_min = 5
ecart_max = 10
nb_cluster = 5
cpu = 5
methode_dist = "euclidean"

def main():
    km = Kmeans(tc.init_board_gauss(nb_points, nb_classe, mini, maxi, ecart_min, ecart_max), nb_cluster=nb_cluster, cpu=cpu, methode_dist=methode_dist, adr=img_dir)
    km.run_global(choose_nb_graph=True, grphq=True)
    km.save(km_path)
    print("\n{}".format(km))
    return None

main()
