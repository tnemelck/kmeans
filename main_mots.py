#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:18:07 2018

@author: elvex
"""


from kmeans import Kmeans
import bdd

path = './BDD/sncf/' #Répertoire des fichiers json de tweets.
img_dir = './IMG/img_sncf_mots'
km_path = './KM/sncf_mots'
drop_var = 5

def main():
    df = bdd.concat_dir(path)
    df = bdd.drop_profile(df, drop_var)
    df = bdd.bdd2bow(df)
    idx, mtx = bdd.df2np(df)
    col = df.columns.values.astype(str)
    del df
    
    km = Kmeans(mtx, nb_cluster=nb_cluster, cpu=cpu, methode_dist=methode_dist, adr=img_dir, index=idx)
    km.run_global(choose_nb_graph=True)
    bdd.print_means_words(km, col)
    km.save(km_path)
    print("\n{}".format(km))
    return None

main()