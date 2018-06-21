#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:18:07 2018

@author: elvex
"""


from kmeans import Kmeans
import bdd

def main():
    path = '/home/elvex/utt/SRT6/IF25/projet/BDD/sncf/'
    img_dir = '/home/elvex/utt/SRT6/IF25/projet/img_sncf_mots'
    km_path = '/home/elvex/utt/SRT6/IF25/projet/KM/sncf_mots'
    drop_var = 5
    
    df = bdd.concat_dir(path)
    df = bdd.drop_profile(df, drop_var)
    df = bdd.bdd2bow(df)
    idx, mtx = bdd.df2np(df)
    del df
    
    km = Kmeans(mtx, nb_cluster=5, cpu=8, methode_dist="cosine", adr=img_dir, index=idx)
    km.run_global(choose_nb_graph=True)
    
    km.save(km_path)
    
    return km