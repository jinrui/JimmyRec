#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Fm
from ..layers.layer_utils import build_input_layers
from ..features.feature_columns import build_feature_columns, dict_from_feature_columns

#改进版DeepFm，和原来的有所不同，更简洁，主要是为了适应一个slot多个值的问题
#比如：用户最近一周听过的歌曲这个特征，是个变成数组，原模型不能很好的支持

def DeepFm(dnn_feature_columns,line_feature_columns,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
    inputs_list = build_feature_columns(dnn_feature_columns + line_feature_columns)
    print('gaga', inputs_list)
    dnn_fea_dict, num_fea_dict = dict_from_feature_columns(inputs_list, dnn_feature_columns + line_feature_columns)
    line_input = tf.concat(num_fea_dict.values(), axis = 1)
    #dnn_fea_dict 可能是多值特征,所以有一个reduce_mean
    dnn_fea_list = [tf.reduce_mean(fea) for fea in dnn_fea_dict.values()]
    print('hehe',dnn_fea_list, dnn_fea_list[0].shape)
    print(num_fea_dict.values())
    dnn_input_fea = tf.concat(dnn_fea_list + num_fea_dict.values(), axis = 1)
    lr_logit = Dense(1)(line_input) #lr
    fm_input_fea = tf.reshape(dnn_input_fea, [-1, len(dnn_fea_list), dnn_fea_list[0].embedding_dim])
    fm_logit = Fm()(fm_input_fea) #fm
    blocks = ks.models.Sequential(name='dynamic-blocks')
    for hit in dnn_hidden_units:
        blocks.add(Dense(hit))
        blocks.add(Activation(dnn_activation_fn))
        if  dnn_dropout is not None:
            blocks.add(Dropout(dnn_dropout))
    deep_logit = blocks(dnn_input_fea) #dnn
    all_concat = tf.concat([lr_logit, fm_logit, deep_logit], axis=1)
    output = Dense(n_classes, Activation(output_activation))(all_concat)
    myinputs = inputs_list.values()
    model = tf.keras.models.Model(inputs=myinputs, outputs=output)
    return model


def DeepFm_v2(dnn_feature_columns,line_feature_columns,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
    dnn_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=dnn_feature_columns + line_feature_columns, name="dnn_input_layer")
    fm_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=dnn_feature_columns , name="fm_input_layer")
    line_input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=line_feature_columns, name="line_input_layer")
    inputs_list = build_input_layers(dnn_feature_columns + line_feature_columns)
    

    lr_logit = Dense(1)(line_input_layer(inputs_list)) #lr
    dnn_input_fea = dnn_input_layer(inputs_list)
    fm_input_fea = fm_input_layer(inputs_list)
    fm_input_fea = tf.reshape(fm_input_fea, [-1, len(dnn_feature_columns), dnn_feature_columns[0].dimension])
    fm_logit = Fm()(fm_input_fea) #fm
    blocks = ks.models.Sequential(name='dynamic-blocks')
    for hit in dnn_hidden_units:
        blocks.add(Dense(hit))
        blocks.add(Activation(dnn_activation_fn))
        if  dnn_dropout is not None:
            blocks.add(Dropout(dnn_dropout))
    deep_logit = blocks(dnn_input_fea) #dnn
    all_concat = tf.concat([lr_logit, fm_logit, deep_logit], axis=1)
    output = Dense(n_classes, Activation(output_activation))(all_concat)
    myinputs = inputs_list.values()
    model = tf.keras.models.Model(inputs=myinputs, outputs=output)
    return model
