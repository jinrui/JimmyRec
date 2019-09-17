#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from core_layers import DNN_layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os


#经常有多输入，甚至多个id输入，因此id是个input_list,结果为 [(input,input_num, input_dim, output_dim,embeddings_matrix )]
def NCF(user_feas,  user_id_feas, item_feas, item_id_feas, dnn_hidden_units, dnn_dropout=0.1, dnn_activation='relu', user_mf_len= 32,user_mlp_len= 32,item_mf_len= 32,item_mlp_len = 32):
    all_item_feas = [item_feas]
    all_user_feas = [user_feas]
    inputs_list = [item_feas,user_feas ]
    for (input_fea,input_num, input_dim, output_dim, embeddings_matrix) in user_id_feas:
        embeddings_matrix_user = embeddings_matrix
        inputs_list.append(input_fea)
        if embeddings_matrix_user is None:
            embeddings_matrix_user = np.random.rand(input_dim + 1, output_dim)
        embedding_user = Embedding(input_dim = input_dim + 1, # 字典长度
                                output_dim = output_dim, # 词向量 长度（100）
                                weights=[embeddings_matrix_user], # 重点：预训练的词向量系数
                                input_length=input_num, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
        user_x = embedding_user(input_fea)
        all_user_feas.append(Flatten()(user_x))
    
    for (input_fea,input_num, input_dim, output_dim, embeddings_matrix) in item_id_feas:
        embeddings_matrix_user = embeddings_matrix
        inputs_list.append(input_fea)
        if embeddings_matrix_user is None:
            embeddings_matrix_user = np.random.rand(input_dim + 1, output_dim)
        embedding_item = Embedding(input_dim = input_dim + 1, # 字典长度
                                output_dim = output_dim, # 词向量 长度（100）
                                weights=[embeddings_matrix_user], # 重点：预训练的词向量系数
                                input_length=input_num, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
        item_x = embedding_item(input_fea)
        all_item_feas.append(Flatten()(item_x))
    all_item_feas = ks.layers.concatenate(all_item_feas)
    all_user_feas = ks.layers.concatenate(all_user_feas)
    item_mf = Dense(item_mf_len , activation=dnn_activation)(all_item_feas)
    item_mlp = Dense(item_mlp_len , activation=dnn_activation)(all_item_feas)
    user_mf = Dense(user_mf_len , activation=dnn_activation)(all_user_feas)
    user_mlp = Dense(user_mlp_len , activation=dnn_activation)(all_user_feas)
    GMF = ks.layers.multiply([user_mf,item_mf],name='GMF')
    mlp_in = ks.layers.concatenate([user_mlp, item_mlp], axis = -1)
    #for dim in dnn_hidden_units:
     #   mlp_in = Dense( dim, activation=dnn_activation )(mlp_in)
     #   mlp_in = tf.keras.layers.Dropout(dnn_dropout)(mlp_in)
    #mlp_in = DNN_layers(dnn_hidden_units)(mlp_in)
    x = ks.layers.concatenate([GMF, mlp_in], axis = -1)
    forward_out = Dense(1, activation=None,name='forward_out')(x)
    
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[forward_out])
    return model


ncf = NCF(Input((53,), dtype='float', name='user_input' ), [(Input((1,), dtype='float', name='user_id_input' ), 1, 100000,32,None)], Input((53,), dtype='float', name='item_input' ), [(Input((10,), dtype='float', name='item_id_input' ), 1, 100000,32,None)], dnn_hidden_units = [64,32,16])
ncf.compile(optimizer = ks.optimizers.Adam(lr = 1e-5),
                loss = "binary_crossentropy",
                metrics=['accuracy']
                )
ncf.summary()
