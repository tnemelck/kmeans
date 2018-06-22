#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:00:58 2018

@author: elvex
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import os
from os.path import abspath
import shutil




class Grphq():
    """Classe servant à l'affichage des graph de kmeans."""
    def __init__(self, nb_cluster, dim_plot, adr, erase = True):
        """
        Entrée : 
            nb_cluster : int
            dim_plot : tuple de 2 éléments indiquant les colonnes à utiliser pour l'affichage
            adr : adresse du répertoire où sauvegarder les images
            erase : si True, efface le répertoire d'images
        Attributs :
            nb_cluster : int
            dim_plot : tuple de 2 éléments indiquant les colonnes à utiliser pour l'affichage
            adr : adresse du répertoire où sauvegarder les images
            erase : si True, efface le répertoire d'images
            color : couleurs à utiliser
        """
        self.dim_plot = dim_plot
        self.adr_img = abspath(adr)
        self.nb_cluster = nb_cluster
        self.color = self.select_color()
        self.erase = erase
        self.init_dir()
    
    
    def select_color(self):
        """Sélectionne aléatoirement assez de couleurs pour afficher les points de n clusters."""
        color = np.random.random((self.nb_cluster, 3))
        return color
        
        
    def plot_graph(self, data, grp, means, etape, zfill = 3):
        """Affiche les points de data et les centres dans un graph, les groupes
        sont représentés par des couleurs différentes."""
        plt.figure()
        means = means.reshape((1, -1)) if len(means.shape) == 1 else means
        for i in range(self.nb_cluster):
            x = data[(grp[:, 1] == i) , self.dim_plot[0]]
            y = data[(grp[:, 1] == i) , self.dim_plot[1]]
            plt.scatter(x, y, label = "Groupe {}".format(i), color = tuple(self.color[i, :]), marker = '+')
            plt.scatter(means[i, self.dim_plot[0]], means[i, self.dim_plot[1]], color = tuple(self.color[i, :]), marker = 'o')
        plt.legend(loc = 'best')
        plt.title("Répartition à l'étape {}".format(etape))
        plt.plot()
        plt.savefig("{}/img_{}.png".format(self.adr_img, str(etape).zfill(zfill)), format="png", bbox_inches = 'tight', dpi = 200)
        plt.show()
        
    def create_gif(self, duration = 0.5):
        """Crée un gif à partir des images du répertoire de stockage."""
        images = []
        filenames = sorted(glob.glob("{}/img*.png".format(self.adr_img), recursive=True))
        for filename in filenames:
            images.append(imageio.imread(filename, format="png"))
        output_file = "{}/animation.gif".format(self.adr_img)
        imageio.mimsave(output_file, images, duration=duration)
      
        
    def init_dir(self):
        """Vide le répertoire de stockage. """
        if self.erase:
            try:
                shutil.rmtree(self.adr_img)
            except FileNotFoundError:
                pass
        try:
            os.makedirs(self.adr_img, mode = 511)
        except:
            pass
        
    
    def plot_crb_err_cluster(self, err):
        """Affiche les graphs de l'évolutions des métriques d'erreurs en fonction du nombre de cluster,
        permet d'avoir une information sur le nombre optimal de clusters."""
        plt.figure()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=2,
                wspace=0.9, hspace=0.9)
        plt.subplot(321)
        plt.plot(err[0], err[1], label = "Erreur absolue totale", color = 'b', marker = 'o')
        plt.xlabel("Nombre de cluster")
        plt.legend(loc = 'best')
        plt.title("Évolution de l'erreur\nen fonction du nombre de cluster")
        plt.subplot(322)
        plt.plot(err[0], err[2], label = "Erreur relative totale", color = 'b', marker = 'o')
        plt.xlabel("Nombre de cluster")
        plt.legend(loc = 'best')
        plt.title("Évolution de l'erreur relative\nen fonction du nombre de cluster")
        plt.subplot(323)
        plt.plot(err[0], err[3], label = "Variance intra-classe totale", color = 'b', marker = 'o')
        plt.xlabel("Nombre de cluster")
        plt.legend(loc = 'best')
        plt.title("Évolution de la variance intra-classe\nen fonction du nombre de cluster")
        plt.subplot(324)
        plt.plot(err[0], err[-4], label = "Log de l'erreur absolue totale", color = 'b', marker = 'o')
        plt.plot(err[0], err[-3], label = "Log de l'erreur standard.", color = 'r', marker = 'o')
        plt.xlabel("Nombre de cluster")
        plt.legend(loc = 'best')
        plt.title("Évolution de l'erreur et de l'erreur\nd'une distribution uniforme\nen fonction du nombre de cluster")
        plt.subplot(325)
        plt.plot(err[0], err[-2], label = "Fossé statistique", color = 'b', marker = 'o')
        plt.legend(loc = 'best')
        plt.title("Évolution du 'gap statistical'\nen fonction du nombre de cluster")
        plt.xlabel("Nombre de cluster")
        plt.subplot(326)
        plt.bar(err[0], err[-1], label = "Fossé statistique comparé", color = 'm')
        plt.legend(loc = 'best')
        plt.title("Évolution du 'gap statistical' comparé\nen fonction du nombre de clusters")
        plt.xlabel("Nombre de cluster")
        plt.savefig("{}/errorOfCluster.png".format(self.adr_img), format="png", bbox_inches = 'tight', dpi = 200)
        plt.show()
        arg = np.where(err[-1] >= 0)[0][0]
        print("Le nombre idéal de cluster est de {}".format(err[0, arg]))
    
    