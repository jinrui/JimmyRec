#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Fm

#改进版DeepFm，和原来的有所不同，更简洁，主要是为了适应一个slot多个值的问题
#比如：用户最近一周听过的歌曲这个特征，是个变成数组，原模型不能很好的支持

def Din(dnn_feature_columns,line_feature_columns,att_fea_pair,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
    inputs_list = build_feature_columns(dnn_feature_columns + line_feature_columns)
    dnn_fea_dict, num_fea_dict = dict_from_feature_columns(inputs_list, dnn_feature_columns + line_feature_columns)
    #dnn_fea_dict 可能是多值特征,所以有一个reduce_mean
    his_feas = []
    all_hist_fea_name = set()
    for fea_pair in att_fea_pair:
        item_fea = dnn_fea_dict[fea_pair[0]]
        user_feas = dnn_fea_dict[fea_pair[1]]
        his_feas.append(DinAttention_layer()([item_fea, user_feas]))
        all_hist_fea_name.add(fea_pair[0])
        all_hist_fea_name.add(fea_pair[1])
    dnn_fea_list = [tf.reduce_mean(fea, axis=1) for fea in dnn_fea_dict.values() if \
            fea not in all_hist_fea_name]
    dnn_input_fea = tf.concat(dnn_fea_list + his_feas + num_fea_dict.values(), axis = -1)
    output = dnn_input_fea
    for i in range(len(dnn_hidden_units)):
        if i < len(dnn_hidden_units) - 1:
            output = Dense(dnn_hidden_units[i], Activation(dnn_activation_fn))(output)
            if dnn_dropout:
                output = Dropout(dnn_dropout)(output)
        else:
            output = Dense(dnn_hidden_units[i], Activation(output_activation))(output)
    myinputs = inputs_list.values()
    model = tf.keras.models.Model(inputs=myinputs, outputs=output)
    return model
