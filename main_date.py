#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:21:20 2018

@author: elvex
"""

from kmeans import Kmeans
import bdd

path = './BDD/sncf/' #Répertoire des fichiers json de tweets.

img_dir = './IMG/img_sncf_date'
km_path = '/KM/sncf_date'

nb_cluster = 5
cpu = 5
methode_dist = "cosine"


def main():
    
    
    df = bdd.date_dir(path)
    idx, mtx = bdd.df2np(df)
    del df
    
    km = Kmeans(mtx, nb_cluster=nb_cluster, cpu=cpu, methode_dist=methode_dist, adr=img_dir, index=idx)
    km.run_global(grphq=True, choose_nb_graph=True)
    
    km.save(km_path)
    print("\n{}".format(km))
    return None

main()