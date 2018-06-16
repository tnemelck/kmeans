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
from glob import glob
from os.path import abspath
from re import split
from math import pi
from numpy import cos, sin
import datetime


def json2pd(adr):
    """
    Convertit un json de tweets en base de donnée panda.
    Entrée : l'adresse du json
    Sortie : la base de donnée panda
    """
    with open(adr, 'r') as f:
        r = f.read()
        bdd = pd.read_json(r, orient = 'records', lines = True)
    bdd = bdd['user'].apply(pd.Series).join(bdd.drop('user', 1),
             how = "left", lsuffix="_profile", rsuffix="_tweet")
    return bdd


def filterBYlanguage(bdd, lan = 'fr'):
    bdd = bdd[(bdd.lang_tweet == lan)]
    return bdd


def keepNdropPD_txt(bdd):
    bdd = bdd.loc[:, ["id_profile", "text"]]
    return bdd


def aggregate_bddFiltered(bdd):
    grp = bdd.groupby("id_profile")
    bdd = grp.agg(["count", lambda x: "\n".join(x)])
    bdd.columns = bdd.columns.droplevel(0)
    bdd = bdd.rename(columns={ bdd.columns[0]: "counting",  bdd.columns[1]: "text"})
    return bdd

def json2bdd_agreg(json):
    return aggregate_bddFiltered(keepNdropPD_txt(filterBYlanguage(json2pd(json))))


#bdd = aggregate_bddFiltered(keepNdropPD_txt(filterBYlanguage(json2pd(file))))
    

def concat_bdd_aggreg(bdd1, bdd2):
    bdd21 = bdd1.counting.add(bdd2.counting, fill_value=0)
    bdd22 =  bdd1.text.add(bdd2.text, fill_value="")
    bdd2 = pd.concat([bdd21, bdd22], axis=1)
    return bdd2


def concat_dir(dirname):
    path = abspath(dirname)
    lst = glob(path+"/*.json")
    bdd = json2bdd_agreg(lst[0])
    for i in range(1, len(lst)):
        try:
            bdd2 = json2bdd_agreg(lst[i])
            bdd = concat_bdd_aggreg(bdd, bdd2)
        except ValueError as e:
            print("Erreur '{}' sur l'étape {}".format(e, i))
            continue
    return bdd


def drop_profile(bdd, n = 2):
    return bdd.loc[bdd["counting"] >= n, "text"]


def bdd2bow(bdd):
    """
    Transforme un Data Frame panda de tweet en base donnée bag of words,
    chaque collonne correspondant à un mot spécifique
    et chaque ligne à un utilisateur, 
    avec comme contenu de la cellule le nombre d'occurence du mot dans le tweet.
    Entrée : le dataframe panda
    Sortie : le dataframe bag of word
    """
    T = bdd["text"] if isinstance(bdd, pd.core.frame.DataFrame) else bdd
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


def dateBDD(bdd):
    dico_month = {1 : 31, 2 : 28, 3 : 31, 4 : 30, 5 : 31, 6 : 30, 7 : 31,
                  8 : 31, 9 : 30, 10 : 31, 11 : 30, 12 : 30}
    bdd = bdd.loc[:, ['id_tweet', 'created_at_tweet']].set_index('id_tweet')
    bdd.created_at_tweet = bdd.created_at_tweet.apply(lambda x: list(map(int, split('[: -]', str(x)))))
    bdd["hour"] = bdd.created_at_tweet.apply(lambda lst: (lst[-3] + lst[-2] / 60 + lst[-1] / (60**2)) * (pi/12))
    bdd["hour_X"] = bdd.hour.apply(cos)
    bdd["hour_Y"] = bdd.hour.apply(sin)
    bdd["day_X"] = bdd.created_at_tweet.apply(lambda x: cos(x[2] * pi / 6))
    bdd["day_Y"] = bdd.created_at_tweet.apply(lambda x: sin(x[2] * pi / 6))
    bdd["dayweek"] = bdd.created_at_tweet.apply(lambda x: datetime.date(x[0], x[1], x[2]).weekday())
    bdd["dayweek_X"] = bdd.dayweek.apply(lambda x: cos(x * 2 * pi / 7))
    bdd["dayweek_Y"] = bdd.dayweek.apply(lambda x: sin(x * 2 * pi / 7))
    bdd["month_X"] = bdd.created_at_tweet.apply(lambda x: cos(x[1] * pi / dico_month[x[2]]))
    bdd["month_Y"] = bdd.created_at_tweet.apply(lambda x: sin(x[1] * pi / dico_month[x[2]]))
    bdd["year"] = bdd.created_at_tweet.apply(lambda x: x[0])
    bdd.drop(labels = ["created_at_tweet", "hour", "dayweek"], axis = 1, inplace = True)
    return bdd


def json2dateBDD(json):
    return dateBDD(filterBYlanguage(json2pd(json)))



def date_dir(dirname):
    path = abspath(dirname)
    lst = glob(path+"/*.json")
    bdd = json2dateBDD(lst[0])
    for i in range(1, len(lst)):
        try:
            bdd2 = json2dateBDD(lst[i])
            bdd = pd.concat([bdd, bdd2], axis=0)
        except ValueError as e:
            print("Erreur '{}' sur l'étape {}".format(e, i))
            continue
    return bdd
    








        
    
