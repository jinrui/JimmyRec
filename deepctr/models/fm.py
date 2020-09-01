#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input,DenseFeatures, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


#使用FM作为召回模型，很经典
class Fm(Model):
    def __init__(self, user_feature_columns, item_feature_columns ,output_activation = tf.nn.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.user_input_layers = [DenseFeatures(feature_columns=column, name=column.name) for column in user_feature_columns]
        self.item_input_layers = [DenseFeatures(feature_columns=column, name=column.name) for column in item_feature_columns]
        self.output_layer = Dense(1, Activation(output_activation))

    def call(self, x, training = None):
        user_parts = [layer(x) for layer in self.user_input_layers]
        item_parts = [layer(x) for layer in self.item_feature_columns]
        users = tf.keras.layers.add(user_parts)
        items = tf.keras.layers.add(item_parts)
        return self.output_layer(tf.matmul(users, items))