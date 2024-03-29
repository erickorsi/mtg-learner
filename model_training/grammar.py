# -*- coding: utf-8 -*-
"""
Created on Mon Sep 08 10:17 2022

@author: https://github.com/erickorsi

Class and functions for NLP and machine learning of grammatical structure.
"""
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from keras.metrics import FalseNegatives, FalsePositives, Recall, Precision, AUC

import string
import nltk
from polyglot.text import Text
import pandas as pd
import numpy as np

def LSTMClassifier(
    classes,
    total_pos,
    hidden_layers,
    nodes,
    loss,
    metrics,
    ignore_empty=True,
    recurrent_dropout=0,
    dropout=0.5,
    activation='tanh',
    recurrent_activation='sigmoid',
    optimizer='rmsprop'
):
    '''
    '''
    # Binary classification can be simplified to a single probability between 0 and 1
    if classes == 2:
        classes = 1
        loss = 'binary_crossentropy'

    # Initiates the LSTM architecture
    model = Sequential()
    model.add(Input(shape=(None,))) # Variable length sentences, flexible input

    if ignore_empty == True:
        model.add(Embedding(
            name = 'Embedding_mask_0',
            input_dim  = total_pos, # Dimension size based on amount of possible POS tags
            output_dim = total_pos,
            mask_zero  = True
        )) # Maintain dimensions, but ignore filler spaces in shorter texts

    # LSTM layers
    if hidden_layers > 1:
        for layer in range(hidden_layers-1):
            model.add(Bidirectional(LSTM(
                name = 'LSTM',
                units = nodes,
                recurrent_dropout = recurrent_dropout,
                dropout = dropout,
                activation = activation,
                recurrent_activation = recurrent_activation,
                return_sequences = True # Hidden layers with stacked LSTM
            )))
    model.add(Bidirectional(LSTM( 
        name = 'LSTM',
        units = nodes,
        recurrent_dropout = recurrent_dropout,
        dropout = dropout,
        activation = activation,
        recurrent_activation = recurrent_activation
    ))) # Final LSTM layer

    # Output layer for classification
    model.add(Dense(
        name='Output',
        units = classes, # Number of outputs
        activation = 'sigmoid' # Normalizes into a probability between 0 and 1
    ))

    # Compile the model
    model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy',Precision(),Recall(),FalseNegatives(),FalsePositives(),AUC()])
    return model

def remove_punctuation(text):
    '''
    '''
    translator = str.maketrans('','', string.punctuation)
    text = text.translate(translator)
    return text

def convert_pos(text, language=None):
    '''
    '''
    text = Text(text, hint_language_code=language)
    pos = []
    for word,tag in text.pos_tags:
        pos.append(tag)
    return pos

def grammar_nlp(text, language=None, ignore_punctuation=False):
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

    return sequence, total_pos

def get_data(dataset, independent_var, texts_var='auto'):
    '''
    '''
    # Gets the defined classes
    Y = np.array(dataset[independent_var])
    # Gets the texts
    if texts_var == 'auto': # If there is only 1 other column in the dataset
        X = dataset.drop(independent_var, axis=1)
    else:
        X = dataset[texts_var]

    # Converts grammar
    for index in X.index:
        text = X[index]
        X[index] = grammar_nlp(text)

    # Fills shorter texts with 0
    X = X.fillna(0).to_numpy()

    return X, Y
