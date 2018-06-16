#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:21:20 2018

@author: elvex
"""

from kmeans import Kmeans
import bdd

def main():
    path = '/home/elvex/utt/SRT6/IF25/projet/BDD/sncf/'
    img_dir = '/home/elvex/utt/SRT6/IF25/projet/img_sncf'
    km_path = '/home/elvex/utt/SRT6/IF25/projet/KM/sncf_date'
    
    df = bdd.date_dir(path)
    idx, mtx = bdd.df2np(df)
    del df
    
    km = Kmeans(mtx, nb_cluster=10, cpu=8, methode_dist="cosine", adr=img_dir, index=idx)
    km.run_global(grphq=True, choose_nb_graph=True)
    
    km.save(km_path)
    
    return km