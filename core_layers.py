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

#DNN
class DNN_layers(Layer):
    def __init__(self, hidden_units, activations, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activations = activations
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN_layers, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernels = []
        self.bias = []
        last_dim = int(input_shape[-1])
        for i in range(len(self.hidden_units)):
            #print (last_dim, dim)
            kernel = self.add_weight(name='kernel_' + str(i),  shape=(last_dim, self.hidden_units[i]), initializer='glorot_uniform',trainable=True)
            self.kernels.append(kernel)
            last_dim = self.hidden_units[i]
            bias = self.add_weight(name='bias_' + str(i),  shape=(self.hidden_units[i],), initializer='zeros',trainable=True)
            self.bias.append(bias)
            self.built = True
        super(DNN_layers, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = False):
        deep_out = x
        for i in  range(len(self.kernels)):
            #x =  ks.layers.dot([x, kernel], axes =(-1,-1) )
            #x = tf.tensordot(deep_out, self.kernels[i],  axes =(-1,0) ) + self.bias[i]
            deep_out = K.dot(deep_out,self.kernels[i])
            deep_out= K.bias_add(deep_out, self.bias[i], data_format='channels_last')
            if  self.activations[i] and self.activations[i] is not None :
                deep_out= tf.keras.layers.Activation(self.activations[i])(deep_out)
            #x= tf.keras.layers.Dropout(self.dropout_rate)(x, training = training)
        return deep_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_units[-1])

#Fm_layer
class Fm_layer(Layer):
    def __init__(self, feat_num , vec_size,  l2_reg=0, **kwargs):
        super(Fm_layer, self).__init__(**kwargs)
        self.l2_reg = l2_reg
        self.feat_num = feat_num
        self.vec_size = vec_size

    def build(self, input_shape):
        self.weight = self.add_weight(name='fm_weight',  shape=(self.feat_num, 1), initializer='glorot_uniform',trainable=True)
        self.bias = self.add_weight(name='fm_bias',  shape=(1,), initializer='glorot_uniform',trainable=True)
        self.fea_vec = self.add_weight(name='fm_vec',  shape=(self.feat_num, self.vec_size), initializer=ks.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=1024)
,trainable=True)
        super(Fm_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = None):
        lr = tf.matmul(x,self.weight ) + self.bias
        square_of_sum = tf.square(K.dot(x,  self.fea_vec))
        sum_of_suqare = K.dot(K.square(x), K.square(self.fea_vec))
        haha = 0.5 * tf.reduce_sum((square_of_sum - sum_of_suqare), axis = 1, keep_dims=True) + lr 
        return   haha#th fm part has error ,make auc 0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


#BiInteraction_layer
class BiInteraction_layer(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(BiInteraction_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        super(BiInteraction_layer, self).build(
            input_shape)  # Be sure to call this somewhere!


    def call(self, x, training = False):
        square_of_sum = tf.square(tf.reduce_sum(x, axis = 1,keepdims = True))
        sum_of_suqare = tf.reduce_sum(tf.multiply(x, x), axis = 1,keepdims = True)
        return 0.5 * (square_of_sum - sum_of_suqare)
        

#LocalActivationUnit_layer
class LocalActivationUnit_layer(Layer):
     def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(LocalActivationUnit_layer, self).__init__(**kwargs)

#SimpleAttention_layer
class SimpleAttention_layer(Layer):
     def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(SimpleAttention_layer, self).__init__(**kwargs)





