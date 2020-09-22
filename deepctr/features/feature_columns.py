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
from tensorflow.python.keras.initializers import RandomNormal

"""
把深度模型的特征定义为三种格式：1、离散->embedding  2、离散->onehot  3、连续->数字
"""
class embedding_column():
    def __init__(self, name,  vocabulary_size,  embedding_dim, max_len = 1,use_hash=False, embeddings_initializer = None, trainable=True, dtype=tf.float32):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.use_hash = use_hash
        self.trainable = trainable
        self.dtype = dtype
        if self.use_hash:
            self.dtype = tf.string
        self.embeddings_initializer = embeddings_initializer
        if self.embeddings_initializer is None:
            self.embeddings_initializer  = RandomNormal(mean=0.0, stddev=0.0001, seed=1991)

class numeric_column():
    def __init__(self, name, num_len = 1, dtype=tf.float32, normfun = None):
        self.name = name
        self.num_len = num_len
        self.dtype = dtype
        self.normfun = None

def build_feature_columns(feature_columns):
    input_dict = {}
    for fc in feature_columns:
        if isinstance(fc, embedding_column): 
            input_dict[fc.name] = Input(
                shape=(fc.max_len,), name=fc.name, dtype=fc.dtype)
            print(fc.name, fc.dtype)
        if isinstance(fc, numeric_column): 
            input_dict[fc.name] = Input(
                shape=(fc.num_len,), name=fc.name, dtype=fc.dtype)
    return input_dict

def dict_from_feature_columns(features, feature_columns):
    dnn_fea_dict = {}
    num_fea_dict = {}
    for fc in feature_columns:
        if isinstance(fc, embedding_column): 
            emb = Embedding(fc.vocabulary_size, fc.embedding_dim, \
                embeddings_initializer=fc.embeddings_initializer,\
                name = 'emb_' + fc.name, input_length = fc.max_len)
            if fc.use_hash:
                idx =  tf.strings.to_hash_bucket_fast(features[fc.name], fc.vocabulary_size,
                                                        name=None)
            else:
                idx = features[fc.name]
            dnn_fea_dict[fc.name] = emb(idx)
        elif isinstance(fc, numeric_column):
            num_fea_dict[fc.name] = features[fc.name]
           
            if fc.normfun is not None:
                 num_fea_dict[fc.name] = fc.normfun(features[fc.name])
            
    return dnn_fea_dict, num_fea_dict
    
def conf_to_featurecolumns(path):
    fea_list = []
    feature_conf = os.path.join(path, 'feature_list.conf')
    feature_map = os.path.join(path, 'feature_map.conf')
    feature_columns = []
    feature_columns_map = {}
    for line in open(feature_conf):
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        lines = line.strip().split(';')
        tmp_map = {}
        for ll in lines:
            ls = ll.split('=')
            tmp_map[ls[0].strip()] = ls[1].strip()
        fea_list.append(tmp_map)
    for fea_col in fea_list:
        fea_class = fea_col['class']
        name = fea_col['name']
        if fea_class == 'numeric_column':
            num_len = 1
            if 'num_len' in fea_col:
                num_len = int(fea_col['num_len'])
            col = numeric_column(name, num_len)
        if fea_class == 'embedding_column':
            col = embedding_column(name, vocabulary_size = int(fea_col['vocabulary_size'])\
                , embedding_dim = int(fea_col['embedding_dim']), use_hash = fea_col['use_hash'] == 'true')
        feature_columns.append(col)
        feature_columns_map[name] = col
    result = []
    for line in open(feature_map):
        lines = line.strip().split(',')
        result.append([feature_columns_map[ll] for ll in lines])
    return result


    



