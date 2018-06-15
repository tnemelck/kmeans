#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:37:55 2018

@author: elvex
"""

class Message():
    
    def __init__(self):
        
        self.m0 = "I don't understand"
        
        self.m11 = "Unable to standardize."
        self.m12 = "Standardization parameters unfound."
        self.m13 = "Unable to standardize.\nStandardization parameters unfound."
        self.m14 = "Standardization completed."
        self.m15 = "Already un-standardized."
        self.m16 = "Already standardized."
        self.m17 = "Un-standardization completed."
        
        self.m21 = "Unable to normalize."
        self.m22 = "Normalization parameters unfound."
        self.m23 = "Unable to normalize.\nNormalization parameters unfound."
        self.m24 = "Normalization completed."
        self.m25 = "Already un-normalized."
        self.m26 = "Already normalized."
        self.m27 = "Un-normalization completed."
        
        self.m31 = "Unable to mean-normalize."
        self.m32 = "Mean-normalization parameters unfound."
        self.m33 = "Unable to mean-normalize.\nMean-normalization parameters unfound."
        self.m34 = "Mean-normalization completed."
        self.m35 = "Already un-mean-normalized."
        self.m36 = "Already mean-normalized."
        self.m37 = "Un-mean-normalization completed."
        
        self.m41 = "Already biased."
        self.m42 = "Already unbiased."
        self.m43 = "Biazation completed."
        self.m44 = "Un-biazation completed."
        
        self.m51 = "Already labellized."
        self.m52 = "Already un-labellized."
        self.m53 = "Labellization completed."
        self.m54 = "Un-labellization completed."
        self.m55 = "Not labellized."
        self.m56 = "This is the label vector."
        
        self.dict = {0 : self.m0,
                     11 : self.m11,
                     12 : self.m12,
                     13 : self.m13,
                     14 : self.m14,
                     15 : self.m15,
                     16 : self.m16,
                     17 : self.m17,
                     21 : self.m21,
                     22 : self.m22,
                     23 : self.m23,
                     24 : self.m24,
                     25 : self.m25,
                     26 : self.m26,
                     27 : self.m27,
                     31 : self.m31,
                     32 : self.m32,
                     33 : self.m33,
                     34 : self.m34,
                     35 : self.m35,
                     36 : self.m36,
                     37 : self.m37,
                     41 : self.m41,
                     42 : self.m42,
                     43 : self.m43,
                     44 : self.m44,
                     51 : self.m51,
                     52 : self.m52,
                     53 : self.m53,
                     54 : self.m54,
                     55 : self.m55,
                     56 : self.m56}
        
        
    def display(self, v=False, m=0):
        if v: 
            print(self.dict.get(m, 0))
        else:
            pass
        
        
        
        
        