#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input,DenseFeatures, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Attention_layer
from ..layers.layer_utils import build_input_layers
from ..features.feature_columns import build_feature_columns, dict_from_feature_columns

def DSSM(user_feature_columns,item_feature_columns1, item_feature_columns2,mode='pointwise',user_dnn_hidden_units=None,item_dnn_hidden_units=None,
    dnn_activation_fn=tf.nn.relu,dnn_dropout=None,output_activation = tf.nn.sigmoid,
    n_classes=1,batch_norm=False):
        inputs_list = build_feature_columns(user_feature_columns + item_feature_columns1 + item_feature_columns2)
        dnn_fea_dict, num_fea_dict = dict_from_feature_columns(inputs_list, user_feature_columns + item_feature_columns1 + item_feature_columns2)
        user_dnn_fea_list = [tf.reduce_mean(dnn_fea_dict[ufc.name], axis=1) for ufc in user_feature_columns if ufc.name in dnn_fea_dict]
        item_dnn_fea_list1 = [tf.reduce_mean(dnn_fea_dict[ifc.name], axis=1) for ifc in item_feature_columns1 if ifc.name in dnn_fea_dict]
        item_dnn_fea_list2 = [tf.reduce_mean(dnn_fea_dict[ifc.name], axis=1) for ifc in item_feature_columns2 if ifc.name in dnn_fea_dict]

        user_lr_fea_list = [num_fea_dict[ufc.name] for ufc in user_feature_columns if ufc.name in num_fea_dict]
        item_lr_fea_list1 = [num_fea_dict[ifc.name] for ifc in item_feature_columns1 if ifc.name in num_fea_dict]
        item_lr_fea_list2 = [num_fea_dict[ifc.name] for ifc in item_feature_columns2 if ifc.name in num_fea_dict]
        user_dnn_input_fea = tf.concat(user_dnn_fea_list + user_lr_fea_list, axis = -1)
        item_dnn_input_fea1 = tf.concat(item_dnn_fea_list1 + item_lr_fea_list1, axis = -1)
        item_dnn_input_fea2 = tf.concat(item_dnn_fea_list2 + item_lr_fea_list2, axis = -1)
        user_blocks = ks.models.Sequential(name='user_dynamic-blocks')
        for hit in user_dnn_hidden_units:
                user_blocks.add(Dense(hit))
                user_blocks.add(Activation(dnn_activation_fn))
        item_blocks = ks.models.Sequential(name='item_dynamic-blocks')
        for hit in item_dnn_hidden_units:
                item_blocks.add(Dense(hit))
                item_blocks.add(Activation(dnn_activation_fn))
        user_output = user_blocks(user_dnn_input_fea)
        item_output1 = item_blocks(item_dnn_input_fea1)
        item_output2 = item_blocks(item_dnn_input_fea2)
        deep_logit1 = user_output * item_output1 #dnn
        deep_logit2 = user_output * item_output2 #dnn
        print(deep_logit1, deep_logit2)
        output1 = Dense(n_classes, Activation(None))(deep_logit1)
        output2 = Dense(n_classes, Activation(None))(deep_logit2)
        #bpr loss: 其实就是log loss的推导,loss = -y_label * log(final_output) = -log(final_output)
        #也可以用 hinger loss，loss = max(0, margin - label * (s1 - s2)) = max(0, margin - s1 + s2)
        #pw 中label都为1
        final_output = output_activation(output1)
        if mode == 'pairwise':
            final_output = output_activation(output1 - output2) 
        myinputs = inputs_list.values()
        model = tf.keras.models.Model(inputs=myinputs, outputs=final_output)
        return model

#使用dssm作为召回模型，很经典
class Dssm_v1(Model):
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