#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.feature_column import embedding_column,indicator_column,bucketized_column,categorical_column_with_vocabulary_file,categorical_column_with_vocabulary_list,categorical_column_with_identity,numeric_column,categorical_column_with_hash_bucket,crossed_column

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os

def build_input_layers(feature_columns):
    input_list = []
    for fc in feature_columns:
        if isinstance(fc, embedding_column): 
            input_list.append(Input(
                shape=(fc.dimension,), name=fc.name, dtype=fc.dtype))
        if isinstance(fc, indicator_column): 
            input_list.append(Input(
                shape=(fc.variable_shape[-1],), name=fc.name, dtype=fc.dtype))
        if isinstance(fc, numeric_column): 
            input_list.append(Input(
                shape=(fc.shape[0],), name=fc.name, dtype=fc.dtype))
        return input_list
