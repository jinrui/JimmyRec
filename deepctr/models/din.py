#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Fm

#deep interest network. 利用attention机制找到和item最想相似的特征
def Din(dnn_feature_columns,line_feature_columns,hist_feature_name,target_feature_name,dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
        inputs_list = build_feature_columns(dnn_feature_columns + line_feature_columns)
        dnn_fea_dict, num_fea_dict = dict_from_feature_columns(inputs_list, dnn_feature_columns + line_feature_columns)
        key_feature_columns = []
        query_feature_columns = []
        for fc in dnn_feature_columns:
                if fc in hist_feature_name:
                        key_feature_columns.append(fc)
                elif fc in target_feature_name:
                        query_feature_columns.append(fc)
        key_features = tf.concat(key_feature_columns, axis = -1)
        query_features = tf.concat(query_feature_columns, axis = -1)
        att_feature = Attention_layer([128, 64, 1], 'din')(query_features, [key_features, key_features])
    #dnn_fea_dict 可能是多值特征,所以有一个reduce_mean
        dnn_fea_list = [tf.reduce_mean(fea, axis=1) for fea in dnn_fea_dict.values()]
        dnn_input_fea = tf.concat(dnn_fea_list + [att_feature] + num_fea_dict.values(), axis = -1)
        blocks = ks.models.Sequential(name='dynamic-blocks')
        for hit in dnn_hidden_units:
                blocks.add(Dense(hit))
                blocks.add(Activation(dnn_activation_fn))
        if  dnn_dropout is not None:
                blocks.add(Dropout(dnn_dropout))
        deep_logit = blocks(dnn_input_fea) #dnn
        print(deep_logit)
        output = Dense(n_classes, Activation(output_activation))(deep_logit)
        myinputs = inputs_list.values()
        model = tf.keras.models.Model(inputs=myinputs, outputs=output)
        return model
