# -*- coding: utf-8 -*-
"""
Created on Mon Sep 08 10:17 2022

@author: https://github.com/erickorsi

Class and functions for NLP and machine learning of grammatical structure.
"""
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.metrics import FalseNegatives, FalsePositives, Recall, Precision, AUC

import string
import nltk
from polyglot.text import Text
import pandas as pd
import numpy as np

class LSTM():
    '''
    '''

def remove_punctuation(text):
    '''
    '''
    translator = str.maketrans('','', string.punctuation)
    text = text.translate(translator)
    return text

def convert_pos(text, language='en'):
    '''
    '''
    text = Text(text, hint_language_code=language)
    pos = []
    for word,tag in text.pos_tags:
        pos.append(tag)
    return pos

def grammar_nlp(text, language='en', ignore_punctuation=False):
    '''
    '''
    # Dictionary of Parts-of-Speech supported by polyglot
    polyglot_pos = {
        'ADJ':1,'ADP':2,'ADV':3,'AUX':4,'CONJ':5,
        'DET':6,'INTJ':7,'NOUN':8,'NUM':9,
        'PART':10,'PRON':11,'PROPN':12,'SCONJ':13,
        'SYM':14,'VERB':15,'X':16, 'PUNCT':17
    }
    # In case of ignoring punctuation
    if ignore_punctuation == True:
        polyglot_pos.pop('PUNCT')
        text = remove_punctuation(text)

    total_pos = len(polyglot_pos)

    # Get POS values for all words in the text
    pos = convert_pos(text, language)
    sequence = np.array([polyglot_pos[word] for word in pos])