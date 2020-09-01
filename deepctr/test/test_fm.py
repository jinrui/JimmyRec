#!/usr/bin/python
# -*- coding: utf-8 -*-  
from __future__ import print_function
import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from sklearn import cross_validation as cv
from sklearn import preprocessing

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os
from papers_code.core_layers import Fm_layer,DNN_layers


filetxt = np.loadtxt('papers_code/data/ml-100k/sample_fea.txt', delimiter='\t',skiprows=1)
train_data, test_data = cv.train_test_split(filetxt, test_size=0.3)
train_data_y = train_data[:,0]
train_data_x = train_data[:,1:]
print(train_data_x)
train_data_x =  preprocessing.MinMaxScaler().fit_transform(train_data_x)
test_data_y = test_data[:,0]
test_data_x = test_data[:,1:]
print(train_data_x)
train_data_y = np.where(train_data_y >3,1.0,0.0)
test_data_y = np.where(test_data_y >3,1.0,0.0)

#print(train_data_y)
fm = Fm_layer(55, 15)
fm_in =  Input((55,), dtype='float', name='fm_in' )

forward_out = fm(fm_in)
#forward_out = tf.keras.layers.Activation('relu')(forward_out)

forward_out = tf.keras.layers.Activation(tf.nn.sigmoid)(forward_out)
# dnn_layer = DNN_layers([32,1], ['relu', 'sigmoid'])
# forward_out= dnn_layer(fm_in)
#forward_out = tf.keras.layers.Activation(tf.nn.sigmoid)(forward_out)
model = Model(inputs=[fm_in], outputs=[forward_out])
# model = Sequential([
#    Dense(32, input_shape=(55,)),
#    Activation('relu'),
#    Dense(1),
#    Activation('sigmoid'),
# ])
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(0.00001),loss="mse")


from sklearn.metrics import roc_auc_score

batch_print_callback = ks.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch,logs: print(roc_auc_score(train_data_y ,model.predict(train_data_x))))

model.fit(x=train_data_x,
        y=train_data_y,
        batch_size=64, epochs=100,verbose=1),callbacks=[batch_print_callback])
#model.fit(x_train, y_train, epochs = 1000, batch_size=32)
#print(model.evaluate(x={'user_input': test_x_user, 'user_id_input': test_x_user_id, 'item_input': test_x_item, 'item_id_input': test_x_item_id},y = test_y , batch_size=32))

