#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:46:20 2018

@author: elvex
"""

import tweepy
import json
import os

from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream


adr_credentials = "./.credentials"
with open(adr_credentials, mode = 'r') as f : 
    [consumer_key, consumer_secret, access_token, secret_token] = (f.read()).splitlines() 
#Les crédentials de l'api tweeter sont spécifiés dans ce fichier (adr_credentials).
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, secret_token)
api = tweepy.API(auth)


def store_json_timeline(adr, n=10):
    """
    Enregistre les tweets propre à la timeline du profil utilisé.
    Entrées :
        adr : str indiquant le chemin du fichier où enregistrer le json de tweets
        n : limite de tweet à enregistrer
    """
    global api
    init_file(adr)
    for status in tweepy.Cursor(api.home_timeline).items(n):
        # Process a single status
        with open(adr, 'w') as outfile: json.dump(status._json, outfile)


def store_json_followers(adr, n=10):
    """
    Enregistre les tweets propre aux followers du profil utilisé.
    Entrées :
        adr : str indiquant le chemin du fichier où enregistrer le json de tweets
        n : limite de tweet à enregistrer
    """
    global api
    init_file(adr)
    for friend in tweepy.Cursor(api.friends).items(n):
        with open(adr, 'w') as outfile: json.dump(friend._json, outfile)
        
        
def store_json_ownTweet(adr, n=10):
    """
    Enregistre les tweets postés par le profil utilisé.
    Entrées :
        adr : str indiquant le chemin du fichier où enregistrer le json de tweets
        n : limite de tweet à enregistrer
    """
    global api
    init_file(adr)
    for tweet in tweepy.Cursor(api.user_timeline).items(n):
        with open(adr, 'w') as outfile: json.dump(tweet._json, outfile)
    
    
class MyListener(StreamListener):
    """Classe d'écoute des tweets, spécifiant quoi faire avec ces tweets.
    Attributs : 
        adr : str du chemin où sauvegarder les tweets
        lim : nombre de tweets à sauvegarder
        count : nombre de tweets enregistrés
    Méthode :
        on_data : enregistre les tweets écoutés (forme json)
    """
    def __init__(self, adr, lim = 1000):
        self.adr = adr
        self.lim = lim
        self.count = 0
    
    
    def on_data(self, data):
        if self.count <= self.lim: 
            try:
                with open(self.adr, 'a') as f:
                    f.write(data)
                self.count += 1
                print("{} tweets enregistrés.".format(self.count))
                return True
            except BaseException as e:
                print("Error on_data: %s" % str(e))
            return True
 
    
    def on_error(self, status):
        print(status)
        return True


def stream_filter(file_path, lst_mots_cle, lst_lang = ["fr"], lim = 1000):
    """
    Récupère les tweets postés comportant un mot appartenant à une liste de mots clés 
    et les enregistre.
    Entrée :
        file_path : str du chemin où sauvegarder les tweets
        lst_mots_cle : liste de mots clés à filtrer
        lst_lang : liste des langues des tweets à filtrer
        lim : nombre limite de tweets
    """
    global auth
    init_file(file_path)
    if isinstance(lst_mots_cle, str) : lst_mots_cle = lst_mots_cle.split(' ')
    twitter_stream = Stream(auth, MyListener(file_path, lim))
    twitter_stream.filter(track=lst_mots_cle, languages=lst_lang)
    
    
def init_file(adr, blank = False):
    """
    Crée le répertoire et le fichier spécifié par la str adr.
    """
    path = os.path.dirname(adr)
    if not os.path.isabs(path): path = os.path.abspath(path)
    if not os.path.isdir(path): os.makedirs(path)
    if blank: 
        open(adr, 'w').close() 
    else :  
        open(adr, 'a').close()
    return None

    
    

        
""" 
Attributs du dict :
text: the text of the tweet itself
created_at: the date of creation
favorite_count, retweet_count: the number of favourites and retweets
favorited, retweeted: boolean stating whether the authenticated user (you) have favourited or retweeted this tweet
lang: acronym for the language (e.g. “en” for english)
id: the tweet identifier
place, coordinates, geo: geo-location information if available
user: the author’s full profile
entities: list of entities like URLs, @-mentions, hashtags and symbols
in_reply_to_user_id: user identifier if the tweet is a reply to a specific user
in_reply_to_status_id: status identifier id the tweet is a reply to a specific status
"""

