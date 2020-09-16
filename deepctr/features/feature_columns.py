#!/usr/bin/python
# -*- coding: utf-8 -*-  

#import tensorflow as tf 
#import tensorflow.keras as ks

import numpy as np
import tensorflow as tf 
import pandas as pd 
import os
import re
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization

"""
把深度模型的特征定义为三种格式：1、离散->embedding  2、离散->onehot  3、连续->数字
"""
class embeedding_columns():
    def __init__(name,  vocabulary_size,  embedding_dim, max_len = 1,use_hash=False, embeddings_initializer = None, trainable=True, dtype=tf.float32):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.use_hash = use_hash
        self.trainable = trainable
        self.dtype = dtype
        self.embeddings_initializer = embeddings_initializer
        if self.embeddings_initializer is None:
            self.embeddings_initializer  = RandomNormal(mean=0.0, stddev=0.0001, seed=1991)

class numeric_columns():
    def __init__(name, num_len = 1, dtype=tf.float32, normfun = None):
        self.name = name
        self.num_len = num_len
        self.dtype = dtype
        self.normfun = None

def build_feature_columns(feature_columns):
    input_dict = {}
    for fc in feature_columns:
        if isinstance(fc, embeedding_columns): 
            input_dict[fc.name] = Input(
                shape=(fc.max_len,), name=fc.name, dtype=fc.dtype)
        if isinstance(fc, numeric_columns): 
            input_dict[fc.name] = Input(
                shape=(fc.num_len,), name=fc.name, dtype=fc.dtype)
    return input_dict

def dict_from_feature_columns(features, feature_columns):
   dnn_ fea_dict = {}
   num_fea_dict = {}
    for fc in feature_columns:
        if isinstance(fc, embeedding_columns): 
            emb = Embedding(fc.vocabulary_size, fc.embedding_dim, \
                embeddings_initializer=fc.embeddings_initializer,\
                name = 'emb_' + fc.name, input_length = fc.max_len)
            if fc.use_hash:
                idx =  tf.string_to_hash_bucket_fast(features[fc.name], fc.vocabulary_size,
                                                        name=None)
            else:
                idx = features[fc.name]
            fea_dict[fc.name] = tf.nn.embedding_lookup(emb, idx)
        elif isinstance(fc, numeric_columns):
            num_fea_dict[fc.name] = features[fc.name]
            if fc.normfun is not None:
                 num_fea_dict[fc.name] = fc.normfun(features[fc.name])
            
    return dnn_ fea_dict, num_fea_dict
    



