#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os

#DNN
class DNN_layers(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
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
            bias = self.add_weight(name='bias_' + str(i),  shape=(self.hidden_units[i],), initializer='glorot_uniform',trainable=True)
            self.bias.append(bias)
        super(DNN_layers, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = False):
        deep_out = x
        for i in  range(len(self.kernels)):
            #x =  ks.layers.dot([x, kernel], axes =(-1,-1) )
            x = tf.tensordot(deep_out, self.kernels[i],  axes =(-1,0) ) + self.bias[i]
            x = tf.keras.layers.Activation(self.activation)(x)
            x= tf.keras.layers.Dropout(self.dropout_rate)(x, training = training)
            deep_out = x
        return deep_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_units[-1])

#Fm_layer
class Fm_layer(Layer):
     def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(Fm_layer, self).__init__(**kwargs)


#BiInteraction_layer
class BiInteraction_layer(Layer):
     def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(Fm_layer, self).__init__(**kwargs)

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





