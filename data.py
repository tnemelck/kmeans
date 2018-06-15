#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:17:20 2018

@author: elvex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:48:43 2018

@author: elvex
"""

import numpy as np
from message import Message

class Data:
    
    """
    Classe qui permet de représenter un dataset avec matrice de données, index, biais et label.
    Permet aussi la normalisation et la standardization des données.
    """
    
    def __init__(self, data, standardize = False, normalize = False, mean_normalize = False,
                 biased = False, labelize = False, lbl = 0, index = None, verbose = True):
        """
        Fonction d'initialisation de la classe Data.
        Paramètres d'entrées :
            data : numpy matrice de données brut (non labelisée, non biaisée, etc)
            standardize : booléen indiquant si la matrice doit être standardizée
            normalize : booléen indiquant si la matrice doit être normalisée
            mean_normalize : booléen indiquant si la matrice doit être mean-normalisée
            biased : normalize : booléen indiquant si la matrice doit biaisée (ajout d'une colonne de 1)
            labelize : booléen indiquant si un vecteur label doit être ajoutée
            lbl : numpy vecteur des labels
            index : numpy vecteur d'index
            
        Attributs de classe :
            data : numpy matrice de données
            moy : moyenne du dataset brut
            e : écart-type du dataset brut
            min: vecteur des valeurs minimums de la matrice de données brut, par colonne
            max: vecteur des valeurs maximums de la matrice de données brut, par colonne
            _is_standardized : indique si data est actuellement standardizée
            _is_normalized : indique si data est actuellement normalisée
            _is_mean_normalized : indique si data est actuellement mean-normalisée
            _is_biased : indique si data est actuellement biaisée
            msg : instance de la classe Message qui contient juste des messages à afficher
            index : vecteur d'index
            
            
        """
        self.data = np.array(data)
        self.moy, self.e = None, None
        self.min, self.max = None, None
        self._is_standardized = False
        self._is_normalized = False
        self._is_mean_normalized = False
        self._is_biased = False
        #self._is_labelled = False
        self.msg = Message()
        self.init_standardize_var()
        self.init_normalize_var()
        self.index = index if np.any(index != None) else np.arange(self.data.shape[0])
        self.index = self.index.reshape((-1,1))
        if standardize: self.standardize(verbose)
        if normalize: self.normalize(verbose)
        if mean_normalize: self.mean_normalize(verbose)
        if biased: self.biase(verbose)
        if lbl: self.lbl = np.array(lbl)
        #if lbl: self.labelise(lbl)
        
        
    def _deb(self):
        """Renvoie l'indice de début de la matrice non biaisée data, 1 si elle est biaisé, 0 sinon. """
        deb = int(self._is_biased)
        return deb
        
    
    def get_standardize_var(self):
        """Calcule et renvoie la moyenne et l'écard type de la matrice data, par colonne.
        Sortie : moyenne, ecart-type"""
        deb = self._deb()
        moy = np.nanmean(self.data[:, deb:], axis=0).reshape(1, self.data.shape[1])
        e = np.nanstd(self.data[:, deb:], axis=0).reshape(1, self.data.shape[1])
        return (moy, e)
    
    
    def set_standardize_var(self, moy, e):
        """Attribue moy et e.
        Entrée : moy, e"""
        self.moy, self.e = moy, e
        
        
    def init_standardize_var(self):
        """Calcule et attribue les variables de standardisation, moy et e."""
        moy, e = self.get_standardize_var()
        self.set_standardize_var(moy, e)
        
        
    def get_normalize_var(self):
        """Calcule et renvoie le minimum et le maximum de la matrice data, par colonne.
        Sortie : minimum, et maximum"""
        deb = self._deb()
        mini, maxi = np.nanmin(self.data[:, deb:], axis = 0), np.nanmax(self.data[:, deb:], axis = 0)
        return (mini, maxi)
    
    
    def set_normalize_var(self, mini, maxi):
        """Attribue min et max.
        Entrées : min, max"""
        self.min, self.max = mini, maxi
        
        
    def init_normalize_var(self):
        """Calcule et attribue les variables de normalisation, min et max."""
        mini, maxi = self.get_normalize_var()
        self.set_normalize_var(mini, maxi)
        
        
    def standardize(self, verbose = True):
        """Standardise la matrice data si elle ne l'est pas déjà.
        data_standard = (data - moy) / ecart-type
        Les données sont centrée réduites en 0."""
        
        test = self.default_state(verbose = verbose)
        if not test:
            self.msg.display(verbose, 11)
        elif np.any(self.moy == None) or np.any(self.e == None) : 
            if verbose: 
                self.msg.display(verbose, 13)
        else:
            deb = self._deb()
            self.data[:, deb:] = (self.data[:, deb:] - self.moy) / self.e
            self._is_standardized = True
            self.msg.display(verbose, 14)
        
        
    def unStandardize(self, verbose = True):
        """Dé-standardise la matrice data si elle l'est.
        data = (data_standard * ecart-type) + moyenne
        Retourne Faux si l'un des paramètres est manquant, Vrai sinon."""
        if self._is_standardized:
            if np.any(self.moy == None) or np.any(self.e == None) :
                self.msg.display(verbose, 12)
                return False
            else:
                deb = self._deb()
                self.data[:, deb:] =  (self.data[:, deb:] * self.e) + self.moy
                self._is_standardized = False
                self.msg.display(verbose, 17)
                return True
        else:
            self.msg.display(verbose, 15)
            return True
                
                
    def normalize(self, verbose = True):
        """Normalise la matrice data si elle ne l'est pas déjà.
        data_standard = (data - min) / (max - min)
        Les données sont comprises entre 0 et 1"""
        test = self.default_state(verbose = verbose)
        if not test:
            self.msg.display(verbose, 21)
        elif np.any(self.min == None) or np.any(self.max == None) : 
            self.msg.display(verbose, 23)
        else:
            deb = self._deb()
            self.data[:, deb:] = (self.data[:, deb:] - self.min) / (self.max - self.min)
            self._is_normalized = True
            self.msg.display(verbose, 24)
        
        
    def unNormalize(self, verbose = True):
        """Dé-normalise la matrice data si elle l'est.
        data = (data_standard * (max - min)) + min
        Retourne Faux si l'un des paramètres est manquant, Vrai sinon."""
        if self._is_normalized:
            if np.any(self.min == None) or np.any(self.max == None):
                self.msg.display(verbose, 22)
                return False
            else:
                deb = self._deb()
                self.data[:, deb:] =  (self.data[:, deb:] * (self.max - self.min)) + self.min
                self._is_normalized = False
                self.msg.display(verbose, 27)
                return True
        else:
            self.msg.display(verbose, 25)
            return True
        
                
    def mean_normalize(self, verbose = True):
        """Mean-normalise la matrice data si elle ne l'est pas déjà.
        data_standard = (data - moy) / (max - min)
        Les données sont comprises entre -1 et 1 centrée sur la moyenne."""
        test = self.default_state(verbose = verbose)
        if not test:
            self.msg.display(verbose, 31)
        elif np.any(self.min == None) or np.any(self.max == None) or np.any(self.moy == None) : 
            self.msg.display(verbose, 33)
        else:
            deb = self._deb()
            self.data[:, deb:] = (self.data[:, deb:] - self.moy) / (self.max - self.min)
            self._is_mean_normalized = True
            self.msg.display(verbose, 34)
        
        
    def unMean_normalize(self, verbose = True):
        """Dé-mean-normalise la matrice data si elle l'est.
        data = (data_standard * (max - min)) + moy
        Retourne Faux si l'un des paramètres est manquant, Vrai sinon."""
        if self._is_mean_normalized:
            if np.any(self.min == None) or np.any(self.max == None):
                self.msg.display(verbose, 32)
                return False
            else:
                deb = self._deb()
                self.data[:, deb:] =  (self.data[:, deb:] * (self.max - self.min)) + self.moy
                self._is_mean_normalized = False
                self.msg.display(verbose, 37)
                return True
        else:
            self.msg.display(verbose, 35)
            return True
            
            
    def default_state(self, verbose = False):
        """Retourne la matrice à son état brut, renvoie faux si l'un des 
        paramètres est manquant."""
        test1 = self.unStandardize(verbose = verbose)
        test2 = self.unNormalize(verbose = verbose)
        test3 = self.unMean_normalize(verbose = verbose)
        return (test1 and test2 and test3)
    
    
    def biase(self, verbose = True):
        """Biaise la matrice si elle ne l'est pas déjà (ajout d'une colonne de 1)"""
        if not self._is_biased:
            np.hstack((np.ones((self.data.shape[0], 1)), self.data))
            self._is_biased = True
            self.msg.display(verbose, 43)
        else:
            self.msg.display(verbose, 41)
            
            
    def unBiase(self, verbose = True):
        """Débiaise la matrice si elle l'est. """
        if self._is_biased:
            self.data = self.data[:, 1:]
            self._is_biased = False
            self.msg.display(verbose, 44)
        else:
            self.msg.display(42)
            
            
#    def labelise(self, lbl, verbose = True):
#        if not self._is_labeled:
#            self.data = np.hstack((self.data, (np.ones((self.data.shape[0], 1)) * lbl)))
#            self._is_labelled = True
#            self.msg.display(verbose, 53)
#        else:
#            self.msg.display(verbose, 51)
#    
#    
#    def getLabel(self, verbose = True):
#        if self._is_labelled:
#            self.msg.display(verbose, 56)
#            return self.data[:, -1]
#        else:
#            self.msg.display(verbose, 55)
#
#    
#    def unLabelise(self, verbose = True):
#        if self._is_labelled:
#            lbl = self.getLabel(verbose)
#            self.data = self.data[:, :-1]
#            self._is_labelled = False
#            self.msg.display(verbose, 54)
#            return lbl
#        else:
#            self.msg(verbose, 55)
#            
#            
#    def __add__(self, data2):
#        labelized = self._is_labelled and data2._is_labelled
#        biased = self._is_biased or data2._is_biased 
        
            
            
    