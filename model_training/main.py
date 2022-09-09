# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 20:27 2022

@author: https://github.com/erickorsi

Main script for training text classification models using an ensemble of
grammar model and vocabulary model.
"""

class MTGClassifier():
    '''
    '''
    def fit(x, y, language='', ignore_punctuation=False, cross_validation=0):
        '''
        '''
        # Natural Language Processing of the texts
        x = nlp(x, language)

        # Cross validate
        cv

        # First step based on vocabulary (word2vec NN)
        vocab =

        # Second step based on grammatical structure (LSTM)
        grammar = LSTM(x, y)

        # Return object with models attributes
        return self
