#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import feature_column
from tensorflow.python.feature_column import feature_column_v2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os

def build_input_layers(feature_columns):
    input_list = {}
    for fc in feature_columns:
        if isinstance(fc, feature_column_v2.EmbeddingColumn): 
            input_list[fc.name] = Input(
                shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
        if isinstance(fc, feature_column_v2.IndicatorColumn): 
            input_list[fc.name] = Input(
                shape=(fc.variable_shape[-1],), name=fc.name, dtype=fc.dtype)
        if isinstance(fc, feature_column_v2.NumericColumn): 
            input_list[fc.name] = Input(
                shape=(fc.shape[0],), name=fc.name, dtype=fc.dtype)
    return input_list
