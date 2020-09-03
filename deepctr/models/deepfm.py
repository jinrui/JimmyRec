#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Fm
from ..layers.layer_utils import build_input_layers

#改进版DeepFm，和原来的有所不同，更简洁，主要是为了适应一个slot多个值的问题
#比如：用户最近一周听过的歌曲这个特征，是个变成数组，原模型不能很好的支持
class DeepFm(Model):
    def __init__(self, dnn_feature_columns,line_feature_columns,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False, **kwargs):
        super(DeepFm, self).__init__(**kwargs)
        self.dnn_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=dnn_feature_columns + line_feature_columns, name="dnn_input_layer")
        self.line_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=line_feature_columns, name="line_input_layer")
        self.fm_layers = []
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation_fn = dnn_activation_fn
        self.dnn_dropout = dnn_dropout
        self.n_classes = n_classes
        self.batch_norm = batch_norm
        self.output_activation=output_activation
        #第一层 embedding_layer
        self.line_layer = Dense(1)
        self.fm_layer = Fm()
        self.blocks = ks.models.Sequential(name='dynamic-blocks')
        for hit in dnn_hidden_units:
            self.blocks.add(Dense(hit))
            self.blocks.add(Activation(dnn_activation_fn))
            if  self.dnn_dropout is not None:
                self.blocks.add(Dropout(self.dnn_dropout))
        self.output_layer = Dense(self.n_classes, Activation(self.output_activation))

    def call(self, x, training = None):
        dnn_part = self.dnn_input_layer(x)
        print(dnn_part.shape)
        line_part = self.line_input_layer(x)
        lr_logit = self.line_layer(line_part) #lr
        #fm_logit = self.fm_layer(dnn_part) #fm
        deep_logit = self.blocks(dnn_part) #dnn
        print(lr_logit.shape, deep_logit.shape)
        all_concat = tf.concat([lr_logit, deep_logit], axis=1)
        return self.output_layer(all_concat)

def DeepFm_v2(dnn_feature_columns,line_feature_columns,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
    dnn_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=dnn_feature_columns + line_feature_columns, name="dnn_input_layer")
    line_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=line_feature_columns, name="line_input_layer")
    dnn_inputs_list = build_input_layers(dnn_feature_columns + line_feature_columns)
    line_inputs_list = build_input_layers(line_feature_columns)
    lr_logit = Dense(1)(line_input_layer(line_inputs_list)) #lr
    #fm_logit = self.fm_layer(dnn_part) #fm
    blocks = ks.models.Sequential(name='dynamic-blocks')
    for hit in dnn_hidden_units:
        blocks.add(Dense(hit))
        blocks.add(Activation(dnn_activation_fn))
        if  dnn_dropout is not None:
            blocks.add(Dropout(dnn_dropout))
    deep_logit = blocks(dnn_input_layer(dnn_inputs_list)) #dnn
    all_concat = tf.concat([lr_logit, deep_logit], axis=1)
    output = Dense(n_classes, Activation(output_activation))(all_concat)
    model = tf.keras.models.Model(inputs=dnn_inputs_list.values(), outputs=output)
    return model
