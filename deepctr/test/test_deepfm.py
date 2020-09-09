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
data = gen_movielens_feas("data/ml-100k")
Features,labels = data,data.pop('rating')
deepfm = DeepFm( feature_columns[0], feature_columns[1], [64,32,8], output_activation = None)
deepfm.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="mse", metrics=['mse'], )
history = deepfm.fit(dict(Features), labels.values,
                     batch_size=128, epochs=10, verbose=2, validation_split=0.2, )
deepfm.summary()
print(deepfm.predict(dict(Features)), labels)

