#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:48:11 2018

@author: elvex
"""

from data import Data 
import numpy as np
import scipy.spatial.distance as dst
from functools import partial
import copy as cpy
from graphic import Grphq
#import re
import os
from pathos.pools import ProcessPool as Pool
#from multiprocessing import Pool
import random as rdm
import math
from pickle import dump, load

class Kmeans():
    """
    Classe permettant la classification non supervisée (clustering) de données, de manière non hiérarchique et avec nombre de clusters prédéfinis.
    Implémente différentes versions de l'algorithme k-means.
    """
    def __init__(self, data = np.ones((2,2)), nb_cluster = 4, means = None, cpu = 1,
                 methode_means = "moy", methode_dist = "euclidean", 
                 p = 3, w = 1, dim_plot = (0, 1), adr = "./img_etape/", erase_dir = True,
                 normalize = False, standardize = False, mean_normalize = False,
                 index = None, verbose = False):
        """
        Fonction d'initialisation de la classe k-means.
        
        Paramètres d'entrée :
            data : matrice de données à utiliser, au format matrice numpy, défaut = 1
            normalize : boolean qui indique si la matrice data doit être normalisée, défaut = False
            standardize : boolean qui indique si la matrice data doit être standardizée, défaut = False
            mean_normalize : boolean qui indique si la matrice data doit être normaliser et centrée, défaut = False
            nb_cluster : entier qui indique le nombre de clusters voulu pour la classification, défaut = 4
            means : matrice numpy, indique les centres à utiliser, si déjà définis, défaut = None
            methode_means : string parmi (moy, med, moye_true, med_true), méthode de calcul des centres, défaut = moy
            methode_dist : string parmi (braycurtis, canberra, chebyshev, cityblock, correlation, cosine,
                                         dice, euclidean, hamming, jaccard, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, 
                                         russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, wminkowski, yule), défaut = euclidean
            p : nombre entier symbolisant un exposant, utile pour les distances de minkowski et wminkoski, défaut 3
            w : vecteur symbolisant des poids, utile pour le distance de wminkoski, défaut 1
            dim_plot : tuple 2D indiquant quelles colonnes de caracrtéristiques pour les représentations graphiques, défaut (0,1)
            adr : string d'adresse indiquant le répertoire où stocker les différents fichiers enregistrés, ./img_etape/
            erase_dir : boolean indiquant si le répertoire de stockage doit être vidé, défaut, défaut True
            cpu : nombre de coeur à utiliser dans la parralélisation des différentes fonctions, défaut 1
            verbose : indique si les normalisations doivent être indiqués
            
        Attributs de classes :
            data : instance de la classe Data contenant les données
            L : nombre d'individus de data
            nb_cluster : nombre de cluster
            means : centres
            grp : matrices de la forme [indices; groupes; distances] qui attribue 
                à chaque individu un groupe lié à un centre et la distance à ce centre
            error : somme des distances des individus à leur groupe, caractérise l'erreur à minimiser
            var : somme des variances intra-classes, à minimiser
            migration : nombre de d'individu qui ont changé de groupe depuis l'itération précédente, à minimiser
            same_means : boolean qui indique si les centres sont restés identiques depuis l'itération précédente
            choose _means : fonction qui calcule les centres
            dist : fonction qui calcule les distances
            p et w : paramètres utiles pour certaines distances
            cpu : nombre de coeur à utiliser pour la parallélisation
            grphq : instance de la classe Grphq qui est utile pour les tracés de graph
            
        """
        self.__dico_mean_method = {"moy" : self.choose_means_moy,
                                   "med" : self.choose_means_med,
                                   "moy_true" : self.choose_means_moy_true,
                                   "med_true" : self.choose_means_med_true}
        self.__dico_dist_method = {
                "braycurtis" : dst.braycurtis,
                "canberra" : dst.canberra,
                "chebyshev" : dst.chebyshev,
                "cityblock" : dst.cityblock,
                "correlation" : dst.correlation,
                "cosine" : dst.cosine,
                "dice" : dst.dice,
                "euclidean" : dst.euclidean,
                "hamming" : dst.hamming,
                "jaccard" : dst.jaccard,
                "kulsinski" : dst.kulsinski,
                "mahalanobis" : dst.mahalanobis,
                "matching" : dst.matching,
                "minkowski" : dst.minkowski,
                "rogerstanimoto" : dst.rogerstanimoto,
                "russellrao" : dst.russellrao,
                "seuclidean" : dst.seuclidean,
                "sokalmichener": dst.sokalmichener,
                "sokalsneath" : dst.sokalsneath,
                "sqeuclidean" : dst.sqeuclidean,
                "wminkowski" : dst.wminkowski,
                "yule" : dst.yule
                }
        
        self.data = Data(np.nan_to_num(data), mean_normalize=mean_normalize, verbose = verbose, 
                         standardize=standardize, normalize=normalize, index=index)
        self.L = self.data.data.shape[0]
        self.nb_cluster = nb_cluster
        self.means = means
        self.grp = None
        self.error = -1
        self.var = -1
        self.migration = -1
        self.same_means = False
        self.choose_means = self.__dico_mean_method.get(methode_means, self.choose_means_moy)
        self.dist = self.__dico_dist_method.get(methode_dist, dst.euclidean)
        self.p, self.w = p, w
        if methode_dist == "minkowski" :
            self.dist = partial(dst.minkowski, p = self.p)
        elif methode_dist == "wminkowski":
            self.dist = partial(dst.wminkowski, p = self.p, w = self.w)
        elif methode_dist == "mahalanobis":
            self.dist = partial(dst.mahalanobis, VI = np.linalg.inv(np.cov(self.data.data, rowvar = False)))
        elif methode_dist == "seuclidean":
            self.dist = partial(dst.seuclidean, V = np.var(self.data.data, axis = 0))
        
        self.grphq = Grphq(nb_cluster, dim_plot, adr, erase_dir)
        self.cpu = min(cpu, os.cpu_count())
        
    
    def run(self, loop = 100, grphq = False, pas = 10, duration_gif = 0.5):
        """ 
        Implémentation la plus simple de la méthode des K-means.
        Se décompose comme ceci :
            0) Choix initial des centres et des groupes associés.
            1) Calcul des nouveaux centres, calcul des nouveaux groupes associés
                Itération jusqu'au critère de convergence.
        Paramètres d'entrée :
            loop : entier qui détermine le nombre maximum d'itération avant un arrêt non
                provoqué par le critère de convergence, défaut 100
            grphq : boolean qui indique si les graphes doivent être affichés et enregistrés
            pas : entier qui indique le nombre d'itération à faire avant d'afficher les graphs, inutile si grphq = False
            duration_gif : réel qui caractérise la durée de chaque image dans la production du gif final, inutile si grphq = False
        Paramètre de sortie : erreur de classification finale.
        """
        if grphq: self.grphq.init_dir()
        self.choose_means_initiate()
        i = 0
        backup = (None, None, -1, -1)
        self.calc_grp()
        if grphq:
            self.grphq.plot_graph(self.data.data, self.grp, self.means, 0)
            self.print_meta_data()
        while (self.cond_conv(backup)) and (i < loop):
            i += 1
            backup = self.backup_metadata()
            self.choose_means()
            if ((self.choose_means != self.choose_means_moy_true) 
                and (self.choose_means != self.choose_means_med_true)) :
                self.calc_grp()
            self.migration = np.count_nonzero((self.grp[:, 1] - backup[1][:, 1]))
            self.same_means = np.array_equal(self.means, backup[0])
            if (i % pas == 0) and (i != loop) and grphq:
                self.grphq.plot_graph(self.data.data, self.grp, self.means, i)
                self.print_meta_data()
        if grphq and ((i == loop) or (i % pas != 0)): 
            self.grphq.plot_graph(self.data.data, self.grp, self.means, i)
            self.print_meta_data()
        if grphq: self.grphq.create_gif(duration = duration_gif)
        return self.error
    
    
    def run_N(self, nb_execution = 10, loop = 100, grphq = False, pas = 10, duration_gif = 0.5):
        """
        Exécute N itération de l'algorithme des k-means, et conserve les centres qui produisent le moins d'erreur.
        Chaque itération est produite à partir de centres initiaux aléatoires, donc les résultats sont différents à chaque fois.
        Retourne cette erreur minimale.
        Les paramètres d'entrée sont les même que pour run, avec l'ajout de :
            nb_execution : entier désignant le nombre de calcul de k-means à faire.
        """
        f = partial(self.__k_run, loop = loop, grphq = grphq, pas = pas)
        pool = Pool(self.cpu)
        memory = list(pool.uimap(f, range(nb_execution)))
        pool.close()
        pool.join()
        ind = np.argmin(np.array([m[0] for m in memory]))
        means = memory[ind][1]
        self.means = means
        self.calc_grp()
        if grphq: self.grphq.create_gif(duration = duration_gif)
        del pool
        return memory[ind][0]
    
    
    def run_global(self, loop = 100, grphq = False, duration_gif = 0.5, pas = 1, choose_nb_graph = False, B=10):
        """
        Implémente l'algorithme des global k-means qui calcule incrémentalement 
        la configuration optimale des groupes pour un nombre de clusters donnée.
        L'algotrithme procède comme suit :
            0) On définit le nombre cluster à 1 et on calcule le centre de la matrice de données.
            1) On incrément le nombre de cluster.
            On définit comme centre les centres de l'étape précédente.
            On définit successivement chaque individu de la matrice de données comme dernier centre, on exécute l'algorithme du
                k-means avec chaque lot de centres et on garde le lot de centre qui minimise l'erreur.
            i+1) On réitère l'étape précédente jusqu'à obtenir le bon nombre de groupe.
        !!! Très gourmand en ressources.
        
        Paramètres d'entrée :
            loop : entier définissant le nombre d'itérations au sein des calcule de k-means avant arrêt du calcul, défaut = 100
            grphq : boolean indiquant si les graphes doivent être affichés et enregistrés.
            duration_gif : réel qui caractérise la durée de chaque image dans la production du gif final, inutile si grphq = False
            pas : entier qui détermine l'écart entre chaque individu à tester pour le choix des individus comme centre.
            choose_nb_graph : boolean, affiche un lot de statistiques qui permettent de déterminer le nombre idéal de clusters.
            B : entier qui qui entre en jeu dans le calcul des statistiques évoquées précédemment.
        Paramètre de sortie :
            err : erreur de classification pour le nombre de cluster choisi.
        """
        pool = Pool(self.cpu)
        pool.close()
        pool.join()
        err = []
        n = self.nb_cluster
        self.set_nb_cluster(1)
        self.choose_means_initiate()
        self.calc_grp()
        self.choose_means()
        self.calc_grp()
        means = self.means
        err.append([1, self.error, self.clustering_error_rel(), self.var])
        if grphq : 
            self.grphq.plot_graph(self.data.data, self.grp, self.means.reshape((1, -1)), 1)
        self.print_meta_data()
        print("Fin de l'étape {}".format(1))
        for i in range(2, n + 1):
            self.set_nb_cluster(i)
            pool.restart()
            f = partial(self.__multi_j, loop = loop, means=means)
            s = pool.uimap(f, range(0, self.L, pas))
            pool.close()
            pool.join()
            s = np.array(list(s))
            arg = np.argmin(s[:, 1])
            j = int(s[arg, 0])
            means_cpy = np.vstack((means, self.data.data[j]))
            self.means = means_cpy
            k = 0
            backup = (None, None, -1, -1)
            self.calc_grp()
            while (self.cond_conv(backup)) and (k < loop):
                k += 1
                backup = self.backup_metadata()
                self.choose_means()
                if ((self.choose_means != self.choose_means_moy_true) 
                    and (self.choose_means != self.choose_means_med_true)) :
                    self.calc_grp()
                self.migration = np.count_nonzero((self.grp[:, 1] - backup[1][:, 1]))
                self.same_means = np.array_equal(self.means, backup[0])
            means = self.means
            if grphq : self.grphq.plot_graph(self.data.data, self.grp, self.means, i)
            self.print_meta_data()
            err.append([i, self.error, self.clustering_error_rel(), self.var])
            print("Fin de l'étape {}".format(i))
        err = np.array(err)
        err = err[np.argsort(err[:, 0]), :].T
        if grphq: self.grphq.create_gif(duration = duration_gif)
        if choose_nb_graph: self.grphq.plot_crb_err_cluster(self.gap_stat(err, B))
        del pool
        return err
    
    
    def run_global_automated(self, grphq = False, duration_gif = 0.5, pas = 1, B=10, loop = 100):
        """
        Implémentation modifiée de run_global où le choix du nombre de cluster est déterminé par des statistiques calculés au fur et à mesure.
        Les paramètres sont : grphq, duration_gif, pas, B, loop et correspondent aux définitions évoqués dans run_global.
        Paramètre de sortie : instance idéal de Kmeans.
        """
        pool = Pool(self.cpu)
        pool.close()
        pool.join()
        mini, maxi = np.min(self.data.data, axis = 0), np.max(self.data.data, axis = 0)
        shape = self.data.data.shape
        i=1
        self.set_nb_cluster(i)
        self.choose_means_initiate()
        self.calc_grp()
        self.choose_means()
        self.calc_grp()
        means = self.means
        if grphq : 
            self.grphq.plot_graph(self.data.data, self.grp, self.means.reshape((1, -1)), 1)
        self.print_meta_data()
        gap, var = self.gap_stat_mono(self.error, i, mini, maxi, shape, pool, B)
        cond = True
        km_cpy = self.copy(erase_dir=False)
        print("Fin de l'étape {}".format(i))
        while cond:
            i+=1
            self.set_nb_cluster(i)
            pool.restart()
            f = partial(self.__multi_j, loop = loop, means=means)
            s = pool.uimap(f, range(0, self.L, pas))
            pool.close()
            pool.join()
            s = np.array(list(s))
            arg = np.argmin(s[:, 1])
            j = int(s[arg, 0])
            means_cpy = np.vstack((means, self.data.data[j]))
            self.means = means_cpy
            k = 0
            backup = (None, None, -1, -1)
            self.calc_grp()
            while (self.cond_conv(backup)) and (k < loop):
                k += 1
                backup = self.backup_metadata()
                self.choose_means()
                if ((self.choose_means != self.choose_means_moy_true) 
                    and (self.choose_means != self.choose_means_med_true)) :
                    self.calc_grp()
                self.migration = np.count_nonzero((self.grp[:, 1] - backup[1][:, 1]))
                self.same_means = np.array_equal(self.means, backup[0])
            means = self.means
            gap_f, var_f = self.gap_stat_mono(self.error, i, mini, maxi, shape, pool, B)
            diff = gap - (gap_f - var_f)
            print("Gap statistical (étape {}) : {}".format(i-1, diff))
            if grphq : self.grphq.plot_graph(self.data.data, self.grp, self.means, i)
            self.print_meta_data()
            print("Fin de l'étape {}".format(i))
            if diff >= 0:
                break
            else:
                gap = gap_f
                km_cpy = self.copy(erase_dir=False)
        if grphq: self.grphq.create_gif(duration = duration_gif)
        self = km_cpy.copy(erase_dir=False)
        self.calc_grp()
        print("Le nombre optimal de classes est : {}".format(self.nb_cluster))
        del pool
        return self
    
    
    def get_methode_mean(self):
        """Renvoie la méthode de choix des centres sous forme de string."""
        return self.choose_means.__name__.split("_")[-1]


    def get_methode_dist(self):
        """Renvoie la méthode de calcule des distabces sous forme de string."""
        return self.dist.__name__.split("_")[-1]
    
    
    def copy(self, data = None, index = None, nb_cluster = None, means = None,
                 methode_means = None, methode_dist = None, erase_dir = None,
                 p = None, w = None, dim_plot = None, adr = None, verbose = False,
                 normalize = None, standardize = None, mean_normalize = None):
        """
        Copie une instance de Kmeans, avec la possibilité de modifier à la volée ses attributs.
        Paramètres d'entrée, tous initialisés à None:
            data, nb_cluster, means, methode_means, methode_dist, erase_dir, p, w, dim_plot, adr, 
            normalize, standardize, mean_normalize, verbose.
        Paramètre de sortie : nouvelle instance de Kmeans.
        !!! Ne réalise pas de copie profonde, si des matrices sont copiés, un changement dans l'instance initiale se 
        répercute sur la matrice de l'instance copiée, et réciproquement.
        """
        n = type(None)
        data = data if not isinstance(data, n) else self.data.data
        nb_cluster = nb_cluster if not isinstance(nb_cluster, n) else self.nb_cluster
        means = means if not isinstance(means, n) else self.means
        methode_dist = methode_dist if not isinstance(methode_dist, n) else self.get_methode_dist
        methode_means = methode_means if not isinstance(methode_means, n) else self.get_methode_mean
        p = p if not isinstance(p, n) else self.p
        w = w if not isinstance(w, n) else self.w
        dim_plot = dim_plot if not isinstance(dim_plot, n) else self.grphq.dim_plot
        adr = adr if not isinstance(adr, n) else self.grphq.adr_img
        normalize = normalize if not isinstance(normalize, n)else self.data._is_normalized
        standardize = standardize if not isinstance(standardize, n) else self.data._is_standardized
        mean_normalize = mean_normalize if not isinstance(mean_normalize, n) else self.data._is_mean_normalized
        erase_dir = erase_dir if not isinstance(erase_dir, n) else self.grphq.erase
        index = index if not isinstance(index, n) else self.data.index
        km = Kmeans(data = data, nb_cluster=nb_cluster, means=means, methode_means=methode_means,
                    methode_dist=methode_dist, p=p, w=w, dim_plot=dim_plot, adr=adr,
                    normalize=normalize, standardize=standardize, mean_normalize=mean_normalize,
                    erase_dir = erase_dir, index=index, verbose = verbose)
        return km
    
    
    def set_nb_cluster(self, nb):
        """Redéfinit le nombre de cluster."""
        self.nb_cluster = nb
        self.grphq.nb_cluster = nb
        self.grphq.color = self.grphq.select_color()
    
        
    def distance(self, vec):
        """Applique la méthode choisi de calcule des distance entre un vecteur
        et les différents centres, et en ressort l'indice du centre le plus
        proche et la distance minimale.
        Paramètre d'entrée :
            vec : numpy vecteur (provenant de la matrice de données)
        Paramètre de sortie :
            [groupe/centre associé, distance à ce centre]
        """
        
        D = np.array([self.dist(vec, v) for v in self.means ])
        result = [np.argmin(D), np.min(D)]
        return result
    
        
    def calc_grp(self):
        """Calcule le vecteur colonne grp qui associe à chaque vecteur ligne
        de data son groupe via un nombre correspondant au numéro du centre le 
        plus proche et la distance à ce centre.
        Modifie les attibuts grp, error et var."""
        grp = np.array([self.distance(vec) for vec in self.data.data])
        self.grp = np.hstack((self.data.index, grp))
        self.error = self.clustering_error()
        self.var = self.var_interclasse()
    
    
    def choose_means_moy(self):
        """Recalcule les centres avec via un calcul de moyenne.
        Modifie l'atribut means."""
        M = self.data.data[np.argwhere(self.grp[:, 1] == 0)[:,0], :]
        means = np.nanmean(M, axis = 0) if M.size > 0 else self.choose_random_mean()
        means = means.reshape((1, -1))
        for i in range(1, self.nb_cluster):
            M = self.data.data[np.argwhere(self.grp[:, 1] == i)[:,0], :]
            m = np.nanmean(M, axis = 0) if M.size > 0 else self.choose_random_mean()
            means = np.vstack((means, m))
        self.means = np.nan_to_num(means)
        
        
    def choose_means_med(self):
        """Recalcule les centres avec via un calcul de médiane.
        Modifie l'atribut means."""
        M = self.data.data[np.argwhere(self.grp[:, 1] == 0)[:,0], :]
        means = np.nanmedian(M, axis = 0) if M.size > 0 else self.choose_random_mean()
        means = means.reshape((1, -1))
        for i in range(1, self.nb_cluster):
            M = self.data.data[np.argwhere(self.grp[:, 1] == i)[:,0], :]
            m = np.nanmedian(M, axis = 0) if M.size > 0 else self.choose_random_mean()
            means = np.vstack((means, m))
        self.means = np.nan_to_num(means)
        
        
    def choose_means_moy_true(self):
        """Recalcule les centres avec via un calcul de moyenne, et associe les centres aux 
        points de la matrice de données la plus proche de ces centres.
        Modifie l'atribut means."""
        self.choose_means_moy()
        self.calc_grp()
        temp = self.grp[self.grp[:, 1] == 0]
        M = int(temp[np.argmin(temp[:, 2]), 0])
        self.means[0, :] = self.data.data[M, :]
        for i in range(1, self.nb_cluster):
            temp = self.grp[self.grp[:, 1] == i]
            M = int(temp[np.argmin(temp[:, 2]), 0])
            self.means[i, :] = self.data.data[M, :]
        self.means = np.nan_to_num(self.means)
        
        
    def choose_means_med_true(self):
        """Recalcule les centres avec via un calcul de médiane, et associe les centres aux 
        points de la matrice de données la plus proche de ces centres.
        Modifie l'atribut means."""
        self.choose_means_med()
        self.calc_grp()
        temp = self.grp[self.grp[:, 1] == 0]
        M = int(temp[np.argmin(temp[:, 2]), 0])
        self.means[0, :] = self.data.data[M, :]
        for i in range(1, self.nb_cluster):
            temp = self.grp[self.grp[:, 1] == i]
            M = int(temp[np.argmin(temp[:, 2]), 0])
            self.means[i, :] = self.data.data[M, :]
        self.means = np.nan_to_num(self.means)
        
        
    def choose_means_initiate(self):
        """
        Méthode de choix aléatoire des nb_cluster premiers centres.
        Modifie l'attribut means.
        """
        smpl = np.random.choice(a = np.arange(self.L), size = self.nb_cluster)
        self.means = self.data.data[smpl, :]
        self.means = np.nan_to_num(self.means)
        
        
    def choose_random_mean(self):
        idx = rdm.randrange(self.L)
        return self.data.data[idx]
    
        
    def __k_run(self, k, loop, grphq, pas):
        """
        Lance une itération de K-means sur une copie de l'instance initiale.
        Retourne l'erreur associée et les centres de l'itération.
        """
        km = self.copy(erase_dir=False)
        e = km.run(loop = loop, grphq=False, pas = pas, )
        memory = (e, km.means)
        if grphq : 
            km.grphq.plot_graph(km.data.data, km.grp, km.means, k)
            km.print_meta_data()
        return memory
    
    def clustering_error(self):
        """
        Calcule l'erreur de classification : la moyenne des distances des individus à leur centre associé.
        Retourne cette erreur.
        """
        S = np.mean(self.grp[:, -1])
        return S
    
    
    def clustering_error_rel(self):
        """Retourne l'erreur relative de classification, qui prend en compte la taille des groupes."""
        s = []
        for i in range(self.nb_cluster):
            g = self.grp[(self.grp[:, 1] == i) , -1]
            s.append((np.sum(g)) / (2 * g.shape[0]))
        s = np.mean(s)
        return s
        
    def var_interclasse(self):
        """Retourne la somme des variances intra-classe."""
        var = np.sum([np.sum(np.var(self.data.data[(self.grp[:, 1] == k) , :], axis=0))
            for k in range(self.nb_cluster)])
        return var
        
    
    def backup_metadata(self):
        """Retourne les centres, les groupes, l'erreur et la variance."""
        return (self.means, self.grp, self.error, self.var)
    
    
    def score(self):
        """ Retourne la variance et l'erreur."""
        return (self.var, self.error)
    
        
    def output_group(self):
        """Retourne les groupes sous la forme : (centre, individus appartenant à ce centre)."""
        result = []
        means_data = cpy.deepcopy(self.data)
        means_data.data = self.means
        means_data.default_state()
        self.data.default_state()
        for i in range(self.nb_cluster):
            t1 = means_data.data[i, :]
            t2 = self.data.index[(self.grp[:, 1] == i)]
            result.append((t1, t2))
        return result
    
    
    def find_group(self, data, graphq = False):
        """Retourne les groupes pour une nouvelle matrice d'individus, sans recalculer les centres."""
        km = self.copy(erase_dir=False)
        km.data = Data(data,
                       mean_normalize=self.data._is_mean_normalized,
                       standardize=self.data._is_standardized,
                       normalize=self.data._is_normalized)
        km.calc_group()
        if graphq: km.plot_graph("de test.")
        result = km.output_group()
        return result
    
    
    def cond_conv(self, backup):
        """
        Définit la condition de convergence en fonction de :
            la modification des centres
            le nombre d'ondividus changeant de groupe
            la variance intra-classe
            la moyenne
        Paramètre d'entrée : une liste backup comme définie dans la fonction éponyme.
        Paramètre de sortie : booléan indiquant si l'algorithme doit se poursuivre.
        """
        cond_means = self.same_means
        cond_migr = self.migration == 0
        cond_var = self.var == backup[3]
        cond_error = self.error == backup[2]
        final_cond = not (True and cond_means and cond_migr and cond_var and cond_error)
        return final_cond
        
        
    
    def print_meta_data(self):
        """Affiche l'erreur de classification la variance intra-classe et le nombre de changement de groupe."""
        print( """ 
              L'erreur de classification vaut : {}
              La variance intra-classe vaut : {}
              Le nombre de variation est : {}
              """.format(self.error, self.var, self.migration))
        
        
    def __multi_j(self, j, means, loop):
        """
        Calcule de k-means à partir de centres prédéfinis.
        
        Paramètres d'entrée :
            j : indice de l'individu à utiliser comme dernier centre
            means : matrice de centres
            loop : nombre d'itération avant fin de calcul des centres
            
        Paramètre de sortie : 
            j : indice de l'individu à utiliser comme dernier centre
            error : erreur associée à ce choix de centre
        """
        means_cpy = np.vstack((means, self.data.data[j]))
        km = self.copy(means = means_cpy, erase_dir=False, verbose = False)
        k = 0
        backup = (None, None, -1, -1)
        km.calc_grp()
        while (km.cond_conv(backup)) and (k < loop):
            k += 1
            backup = km.backup_metadata()
            km.choose_means()
            if ((km.choose_means != km.choose_means_moy_true) 
                and (km.choose_means != km.choose_means_med_true)) :
                km.calc_grp()
            km.migration = np.count_nonzero((km.grp[:, 1] - backup[1][:, 1]))
            km.same_means = np.array_equal(km.means, backup[0])
        return [j, km.error]    
   

    def gap_stat(self, err_init, B = 10):
        """
        Calcule les statistiques utiles dans le choix du nombre optimal de cluster.
        
        Paramètres d'entrée :
            err_init : matrice de la forme [nb_cluster; erreur de classification; erreur relative de classification; variance intra-classe]
            B : Nombre d'itération de k-means avec échantillons aléatoires.
            
        Paramètres de sortie :
            stat : matrice de la forme [nb_cluster; erreur de classification; erreur relative de classification; variance intra-classe;
                                        logarithme de l'erreur de classification; moyenne des log des erreurs avec echantillons aléatoires;
                                        différence entre les logs des erreurs obtenues et la moyenne des log des erreurs avec echantillons aléatoires;
                                        gap statistical | (différence des échantillons n) - (différence des échantillons n+1 * variance des log des erreurs avec echantillons aléatoires)]
        """
        pool = Pool(self.cpu)
        pool.close()
        pool.join()
        mini, maxi = np.min(self.data.data, axis = 0), np.max(self.data.data, axis = 0)
        shape = self.data.data.shape
        log = np.log10(err_init[1])
        mean_alea, var_alea = [], []
        for i in range(1, self.nb_cluster + 1):
            print(0)
            err = []
            f = partial(self.__stat_i, mini = mini, maxi = maxi, shape = shape, i = i)
            pool.restart()
            err = list(pool.map(f, range(B)))
            pool.close()
            pool.join()
            err = np.log10(np.array(err))
            mean_alea.append(np.mean(err))
            var_alea.append(np.std(err))
        mean_alea = np.array(mean_alea)
        var_alea = np.array(var_alea) * np.sqrt(1 + (1/float(B)))
        gap = mean_alea - log 
        diff_gap = gap[0:-1] - (gap[1:] - var_alea[1:])
        diff_gap = np.hstack((diff_gap, 0))
        stat = np.vstack((err_init, log, mean_alea, gap, diff_gap))
        del pool
        return stat
    
    
    def __stat_i(self, k, i, mini, maxi, shape):
        """Calcule l'erreur de classification avec une distribution aléatoire."""
        alea = np.random.uniform(mini, maxi, shape)
        km = self.copy(erase_dir=False, data=alea, nb_cluster=i, normalize=False,
                       standardize=False, mean_normalize = False)
        e = km.run()
        return e
    
    
    def gap_stat_mono(self, err, i, mini, maxi, shape, pool, B=10):
        """
        Donne des statistiques utiles dans le choix du nombre de cluster, pour un nombre de cluster donné.
        
        Paramètres d'entrée :
            err:  scalaire représentant l'erreur de classification pour le nombre de cluster i
            i : nombre de cluster
            mini : valeur minimum de la matrice de données
            maxi : valeur maximale de la mtrice de données
            shape : dimensions de la matrice de données
            pool : permet la parralélisation
            B : Nombre d'itération de k-means avec échantillons aléatoires.
        
        Paramètres de sortie : 
            gap : différence entre les logs des erreurs obtenues et la moyenne des log des erreurs avec echantillons aléatoires
            var_alea : variance des log des erreurs avec echantillons aléatoires
        """
        f = partial(self.__stat_i, mini = mini, maxi = maxi, shape = shape, i = i)
        log = math.log10(err)
        pool.restart()
        err_alea = list(pool.map(f, range(B)))
        pool.close()
        pool.join()
        err_alea = np.log10(np.array(err_alea))
        mean_alea = np.mean(err_alea)
        var_alea = np.std(err_alea) * math.sqrt(1 + (1/float(B)))
        gap = mean_alea - log
        return gap, var_alea
            
    
    
    def __repr__(self):
        """String affichée quand on appelle une instance."""
        s = """Il y a {} clusters. L'erreur de classification vaut : {}.
La méthode de calcule de distance s'appelle {} et la méthode de choix des centres est {}.""".format(self.nb_cluster, self.error, self.get_methode_dist(), self.get_methode_mean())
        return s
    
    
    def __str__(self):
        """String affichée quand on print une instance."""
        s = """Il y a {} clusters. L'erreur de classification vaut : {}.
La méthode de calcule de distance s'appelle {} et la méthode de choix des centres est {}.""".format(self.nb_cluster, self.error, self.get_methode_dist(), self.get_methode_mean())
        return s
            
        
    def save(self, adr):
        d = {"means" : self.means,
             "methode_dist" : self.get_methode_dist(),
             "methode_means" : self.get_methode_mean(),
             "p" : self.p,
             "w" : self.w,
             "nb_cluster" : self.nb_cluster,
             "dim_plot" : self.grphq.dim_plot,
             "adr" : self.grphq.adr_img,
             "erase_dir" : self.grphq.erase}
        dump(d, open(adr, 'wb'))
        return None
    
    
    def load(self, adr):
        d = load(open(adr, 'rb'))
        km = Kmeans(nb_cluster = d["nb_cluster"], means = d["means"], methode_means = d["methode_means"],
                    methode_dist = d["methode_dist"], p = d["p"], w = d["w"], dim_plot = d["dim_plot"],
                    adr = d["adr"], erase_dir = d["erase_dir"])
        return km
        
    
                
                