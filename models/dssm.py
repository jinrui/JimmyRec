#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input,DenseFeatures, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


#使用dssm作为召回模型，很经典
class Dssm(Model):
    def __init__(self, user_feature_columns, item_feature_columns ,dnn_hidden_units=None,\
    dnn_dropout = 0,output_activation = tf.nn.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.user_input_layers = DenseFeatures(feature_columns=user_feature_columns, name=column.name)
        self.item_input_layers = DenseFeatures(feature_columns=item_feature_columns, name=column.name)
        self.output_layer = Dense(1, Activation(output_activation))
        self.user_blocks = ks.models.Sequential(name='user-blocks')
        self.item_blocks = ks.models.Sequential(name='item-blocks')
        for hit in dnn_hidden_units:
            self.user_blocks.add(Dense(hit))
            self.user_blocks.add(Activation(dnn_activation_fn))
            self.user_blocks.add(Dropout(dnn_dropout))
            self.item_blocks.add(Dense(hit))
            self.item_blocks.add(Activation(dnn_activation_fn))
            self.item_blocks.add(Dropout(dnn_dropout))

    def call(self, x, training = None):
        user_parts = [layer(x) for layer in self.user_input_layers]
        item_parts = [layer(x) for layer in self.item_feature_columns]
        users = self.user_blocks(user_parts)
        items = self.item_blocks.add(item_parts)
        return self.output_layer(tf.matmul(users, items))