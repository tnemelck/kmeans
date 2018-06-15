#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:28:05 2018

@author: elvex
"""

from kmeans import Kmeans
import bdd

name_bdd = "./tweet_ramadan.json"


bow = bdd.bdd2bow(bdd.json2pd(name_bdd))
km = Kmeans(data = bow, nb_cluster = 50, cpu = 4, methode_dist = "cosine",
            adr = "./img_ramadan", index = bow.index.values)
#km.run_global(loop = 1000, choose_nb_graph = True)
#km.run(loop = 1000)

def glbl():
    km2 = km.run_global_automated(loop = 1000)
    return km2