#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:43:50 2018

@author: elvex
"""

"""Ensemble de fonctions qui analysent un texte."""


import re
import emot


def formate_txt(txt):
    """
    Transforme un texte en liste contenant les mots et les liens web, 
    sans tenir compte de la ponctuation (hormis #).
    Entrée : txt, une string
    Sortie : Liste de mots et liens
    """
    txt = re.sub(r"(?:https://|http://|wwww)[.\w/…-]+", " ", txt)
    lst = re.findall(r"[\w#-]+", txt.lower())
    return lst


def formate_smiley(txt):
    """
    Utilisent le package emot pour extraire les emoji (inutilisé à l'heure actuelle).
    Entrée : txt, une string
    Sortie : la liste des emoji texte et caractères.
    """
    lst = list(map(lambda x: x["value"], emot.emoji(txt)))
    lst2 = list(map(lambda x: x["value"], emot.emoticons(txt)))
    lst.extend(lst2)
    return lst


def bow(lst_mots):
    """
    Transforme une liste de mots en dictionnaire bag of words qui associe chaque
    mot à son nombre d'occurence dans le texte.
    Entrée : une liste de mots (et de liens)
    Sortie : le dictionnaire bag of words
    """
    d = { s : lst_mots.count(s) for s in set(lst_mots) }
    return d


def tf(lst_mot, dict_bow):
    """
    Renvoie le score tf d'un dictionnaire bag of words compte tenu d'un 
    ensemble de mots prédéfinis : somme des valeurs bag of words des mots
    de la liste de mots prédéfinie.
    Entrée : 
        lst_mots : liste de mots prédéfinie
        dict_bow : dictionnaire bag of words
    Sortie :
        score : l'entier de score
    """
    s = set(map(str.lower, lst_mot))
    score = 0
    for mot in s : score += dict_bow.get(mot, 0)
    return score
    