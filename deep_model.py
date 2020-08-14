#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os

#改进版DeepFm，和原来的有所不同，更简洁，主要是为了适应一个slot多个值的问题
#比如：用户最近一周听过的歌曲这个特征，是个变成数组，原模型不能很好的支持
class DeepFm(Model):
    def __init__(self, feature_columns,dnn_hidden_units=None,feat_num=0,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False,fm_len=10, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=feature_columns, name="DeepFm_input_layer")
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation_fn = dnn_activation_fn
        self.dnn_dropout = dnn_dropout
        self.n_classes = n_classes
        self.batch_norm = batch_norm
        self.fm_len = fm_len
        self.output_activation=output_activation
        self.feat_num=feat_num
        #第一层 embedding_layer
        self.weight = self.add_weight(name='fm_weight',  shape=(self.feat_num, 1), initializer='glorot_uniform',trainable=True)
        self.bias = self.add_weight(name='fm_bias',  shape=(1,), initializer='glorot_uniform',trainable=True)
        self.fea_vec = self.add_weight(name='fm_vec',  shape=(self.feat_num, self.fm_len), initializer=ks.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=1024)
,trainable=True)
        self.blocks = ks.models.Sequential(name='dynamic-blocks')
        for hit in dnn_hidden_units:
            self.blocks.add(Dense(hit))
            self.blocks.add(Activation(dnn_activation_fn))
        self.output_layer = Dense(self.n_classes, Activation(self.output_activation))


    def call(self, x, training = None):
        x = self.input_layer(x)
        lr = tf.matmul(x,self.weight ) + self.bias
        square_of_sum = tf.square(K.dot(x,  self.fea_vec))
        sum_of_suqare = K.dot(K.square(x), K.square(self.fea_vec))
        second_part = 0.5 * (square_of_sum - sum_of_suqare)
        new_weight = tf.reshape(self.fea_vec, shape=[-1, \
                        self.feat_num * self.fm_len])
        dnn_part = self.blocks(new_weight)
        deep_out    = tf.concat([lr, second_part, \
                         dnn_part], axis=1)
        out = self.output_layer(deep_out)
        return   out#th fm part has error ,make auc 0.5

