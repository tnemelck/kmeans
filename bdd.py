#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 02:35:02 2018

@author: elvex
"""

"""Boite à outils de manipulation des base de données de tweets. """

#import json
import pandas as pd
import txt_analysis as TA
from math import log10


def json2pd(adr):
    """
    Convertit un json de tweets en base de donnée panda.
    Entrée : l'adresse du json
    Sortie : la base de donnée panda
    """
    with open(adr, 'r') as f:
        r = f.read()
        bdd = pd.read_json(r, orient = 'records', lines = True)
    return bdd


def bdd2bow(bdd):
    """
    Transforme un Data Frame panda de tweet en base donnée bag of words,
    chaque collonne correspondant à un mot spécifique
    et chaque ligne à un utilisateur, 
    avec comme contenu de la cellule le nombre d'occurence du mot dans le tweet.
    Entrée : le dataframe panda
    Sortie : le dataframe bag of word
    """
    T = bdd["text"]
    T = T.map(TA.formate_txt)
    T = T.map(TA.bow)
    bow = pd.DataFrame.from_dict(T.tolist())
    bow = bow.fillna(0)
    return bow


def filter_bow(bow, mini = 1):
    """
    Permet de filtrer un dataframe bag of words en stipulant un nombre minimum
    de tweets dans lequels les mots doivent apparaître.
    Entrée :
        bow : pandas dataframe bag of words
        mini : entier stipulant le minimum
    Sortie :
        bow_f : le dataframe bag of words filtré
    """
    test = (((bow > 0).sum()) >= mini).as_matrix()
    bow_f = bow.iloc[:, test]
    return bow_f
    


def tf_idf(bow, lst = [], fonction = "idfi"):
    """
    À partir d'un dataframe bag of words, applique une métrique de tf idf pour 
    pondérer le score des mots.
    Entrée :
        bow : dataframe bag of words
        lst : liste de mots à garder dans le dataframe, si nul, tous les mots son gardés
        fonction : fonction de pondération :
            idfn => pas de pondération
            idfi => prend en compte le nombre de tweets et la fréquence d'utilisation des mots
            idfl => comme idfi mais en se laissant une sécurité sur le log10(0)
            idfs => comme idfi mais en se laissant une autre sécurité sur le log10(0)
            idff => prend simplement en compte la fréquence d'utilisation des mots
            idfp => prend en compte le nombre de tweets et la fréquence d'utilisation des mots
    """
    dico = {"idfi" : idfi,
        "idfn" : idfn,
        "idfl" : idfl,
        "idfp" : idfp,
        "idff" : idff,
        "idfs" : idfs}
    D, df = len(bow), (bow > 0).sum()
    f_poids = dico.get(fonction, "idfi")
    idf = bow * f_poids(D, df)
    if len(lst) > 0:  idf = intersection(bow, lst)
    return idf


def intersection(bdd, lst):
    """Renvoie les colonnes d'une bdd pandas qui correspondent aux mots entrés.
    Entrées :
        bdd : panda dataframe
        lst : liste de mots
    Sortie :
        nouvelle dataframe pandas
    """
    s = set(map(str.lower, lst))
    s = s.intersection(set(bdd.columns.values.tolist()))
    return bdd.loc[:, list(s)]


def idfi(D, df):
    return (D/df).apply(log10)


def idfn(D, df):
    return 1


def idfl(D, df):
    return (D/df + 1).apply(log10)


def idff(D, df):
    return 1/df


def idfp(D, df):
    return ((D - df) / df).apply(log10)


def idfs (D, df):
    return (((D + 1) / df).apply(log10)) ** 2

def df2np(df):
    """Convertit un dataframe panda en matrice, renvoie cette matrice et le vecteur d'indice.
    Entrée : 
        df, panda dataframe
    Sortie :
        idx : numpy array des indices de la dataframe
        mtx : numpy array des valeurs de la dataframe
    """
    mtx = df.as_matrix()
    idx = df.index.values
    return (idx, mtx)








        
    
