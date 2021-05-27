#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from ..layers.core_layers import Fm

#通过gate实现多个目标, k experts ,n targets
def MMOE(Model):
    pass
    