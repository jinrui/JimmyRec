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

#另外一种Fm_layer
class Fm(Layer):
    def __init__(self,**kwargs):
        super(Fm, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Fm, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = None):
        print(x.shape)
        if K.ndim(x) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(x)))
        square_of_sum = tf.square(tf.reduce_sum(x, axis = 1, keepdims=True))
        sum_of_suqare = tf.reduce_sum(x * x, axis = 1, keepdims=True)
        return 0.5 * tf.reduce_sum((square_of_sum - sum_of_suqare), axis = 2, keepdims=False) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
    
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

#Attention_layer
class Attention_layer(Layer):
    def __init__(self, hidden_units,mode = 'dot' , activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fc = tf.keras.Sequential()
        for i in range(len(self.hidden_units)):
            self.fc.add(layers.Dense(self.hidden_units[i], activation=self.activation, name="fc_att_"+str(i))) 
        self.fc.add(layers.Dense(1, activation='softmax', name="fc_att_out"))
        super(Attention_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = None):
        query, keys, values = x
        querys = tf.tile(query, mutiples = [1, tf.shape(keys)[1], 1])  #query扩展到和key一样的维度,方便后续计算
        #query key 
        if self.mode = 'dot':
            att = tf.nn.softmax(tf.reduce_mean(querys * keys, axis = -1))
            return tf.reduce_sum(att * values, axis = 1)
        elif self.mode = 'din':
            df1 = tf.concat([keys, keys - querys, querys], axis = -1)
            att = self.fc(df1)
            return tf.reduce_sum(att * values, axis = 1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    #Attention_layer
class MMOE_layer(Layer):
    def __init__(self, expert_hidden_units, gate_hidden_units,num_experts, num_tasks, activation = 'relu', seed=1024, **kwargs):
        self.experts = []
        self.gates = []
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.activation = activation
        super(MMOE_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in range(self.num_experts):
            fc = tf.keras.Sequential()
            for j in range(len(self.expert_hidden_units)):
                fc.add(layers.Dense(self.expert_hidden_units[j], activation=self.activation, name="fc_expert_"+str(i)+"_" + str(j))) 
            self.experts.append(fc)
        for i in range(self.num_tasks):
            fc = tf.keras.Sequential()
            for j in range(len(self.gate_hidden_units)):
                fc.add(layers.Dense(self.gate_hidden_units[j], activation=self.activation, name="fc_gate_"+str(i)+"_" + str(j))) 
            self.gates.append(fc)
        super(MMOE_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training = None):
        experts_out = []
        gates_out = []
        for i in range(self.num_experts):
            experts_out.append(tf.expand_dims(self.experts[i](x), axis = 2))
        for i in range(self.num_tasks):
            gates_out.append(tf.expand_dims(self.gates[i](x), axis = 1))
        experts_out = tf.concat(experts_out, axis = 2) #batch_size * d * num_experts
        gates_out = tf.concat(gates_out, axis = 1) # batch_size * num_tasks * d
        weights_out = tf.matmul(gates_out, experts_out) # batch_size * num_tasks * num_experts
        weights_out = tf.matmul(weights_out, tf.transpose(experts_out, [0, 2, 1])) #batch_size * num_tasks * d
        return weights_out

        


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)





