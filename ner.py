# -*- coding: utf-8 -*-
#    Copyright (C) 2019-TODAY Cleareye.ai(<https://www.cleareye.ai>)
#    Author: Cleareye.ai(<https://www.cleareye.ai>)

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def padding(sentence):
    sentence[2] = pad_sequences(sentence[2], 52, padding='post')
    return sentence


def add_char_information(sentence):
    return [[word, list(str(word))] for word in sentence]


def get_casing(word, case_lookup):
    casing = 'other'

    num_digits = 0
    for char in word:
        if char.isdigit():
            num_digits += 1

    digit_fraction = num_digits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif num_digits > 0:
        casing = 'contains_digit'
    return case_lookup[casing]


def create_tensor(sentence, word2idx, case2idx, char2idx):
    unknownidx = word2idx['UNKNOWN_TOKEN']

    wordindices = []
    caseindices = []
    charindices = []

    for word, char in sentence:
        word = str(word)
        if word in word2idx:
            wordidx = word2idx[word]
        elif word.lower() in word2idx:
            wordidx = word2idx[word.lower()]
        else:
            wordidx = unknownidx
        charidx = []
        for x in char:
            if x in char2idx.keys():
                charidx.append(char2idx[x])
            else:
                charidx.append(char2idx['UNKNOWN'])
        wordindices.append(wordidx)
        caseindices.append(get_casing(word, case2idx))
        charindices.append(charidx)

    return [wordindices, caseindices, charindices]


class Parser:

    def __init__(self, loc = 'models/'):
        # ::Hard coded char lookup ::
        self.char2idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{" \
                 "}!?:;#'\"/\\%$`&=*+@^~|–’•“”…‘><—Ë‐Úéﬀ·àﬁﬂﬄä■ñá¬êö●Іﬃ»○▪➢◆◦®→⧫➔❖✓♦◄►":

            self.char2idx[c] = len(self.char2idx)
        # :: Hard coded case lookup ::
        self.case2idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                         'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.model = load_model(os.path.join(loc, "model.h5"))
        self.model._make_predict_function()
        # loading word2idx
        self.word2idx = np.load(os.path.join(loc, "word2Idx.npy"), allow_pickle=True).item()
        # loading idx2label
        self.idx2label = np.load(os.path.join(loc, "idx2Label.npy"), allow_pickle=True).item()

    def predict(self, sentence):
        sentence = words = word_tokenize(sentence)
        sentence = add_char_information(sentence)
        sentence = padding(create_tensor(sentence, self.word2idx, self.case2idx, self.char2idx))
        tokens, casing, char = sentence
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        prediction = self.model.predict([tokens, casing, char], verbose=False)[0]
        prediction = prediction.argmax(axis=-1)
        prediction = [self.idx2label[x].strip() for x in prediction]
        return list(zip(words, prediction))