#!/usr/bin/python
# -*- coding: utf-8 -*-  
from __future__ import print_function
import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from sklearn import preprocessing

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os
from deepctr.features.feature_utils import gen_movielens_feas,make_featurecolumn
from deepctr.models.deepfm import DeepFm
 
feature_columns = make_featurecolumn('conf/deepfm.conf', 'conf/deepfm.fc')
print(feature_columns)
data = gen_movielens_feas("data/ml-100k")
Features,labels = data,data.pop('rating')
# print(Features)
# print(labels)
# dataset = tf.data.Dataset.from_tensor_slices((dict(Features), labels))
# train_it = dataset.make_one_shot_iterator()
# x_train_it, y_train_it = train_it.get_next()
deepfm = DeepFm(feature_columns[0], feature_columns[1], [64,32,8])
deepfm.build(input_shape=(None, 28, 28, 1))
deepfm.summary()
deepfm.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse')
deepfm.fit(x=Features,
        y=labels,
        batch_size=64, epochs=100,verbose=1)