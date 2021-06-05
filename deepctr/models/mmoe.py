#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import DNN_layers

#通过gate实现多个目标, k experts ,n targets
def MMOE(dnn_feature_columns, num_tasks, tasks_loss, num_experts=4, expert_hidden_units, gate_hidden_units,task_hidden_units
         seed=1024, dnn_activation='relu')
    inputs_list = build_feature_columns(dnn_feature_columns)
    dnn_fea_dict, _ = dict_from_feature_columns(inputs_list, dnn_feature_columns)
    nn_fea_list = [tf.reduce_mean(fea, axis=1) for fea in dnn_fea_dict.values()]
    dnn_input = tf.concat(dnn_fea_list, axis = -1)
    mmoe_layers = MMoELayer(units_experts=units_experts, num_tasks=num_tasks, num_experts=num_experts,
                    name='mmoe_layer')(dnn_input) #task 个 batch_size * 1 * d   数组
    #mmoe_layers ,
    outputs = []
    for i in range(num_tasks):
        tower = DNN_layers(task_hidden_units)(mmoe_layers[i])
        out = Dense(1, activation = tasks_loss[i])(tower)
        outputs.append(out)
    model = tf.keras.models.Model(inputs=inputs_list.values(), outputs=outputs)
    return model


    

    
    