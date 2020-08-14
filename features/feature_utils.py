#!/usr/bin/python
# -*- coding: utf-8 -*-  

import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Layer,Dense,Input, Embedding, LSTM,Bidirectional,Dropout,Activation,Convolution1D, Flatten, MaxPool1D, GlobalAveragePooling1D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
import os
import re

"""
加载特征格式为 label slot_id1:fea_val slot_id2:fea_val slot_id3:fea_val 
slot_id尽量为整数，对应一个特征名，fea_val为特征值，根据自身需要可以是float,string等
返回dataframe，列名为slot_id
"""
def load_jimmysvm(file_name):
    column_name, X, Y = [], [],[]
    for line in open(file_name):
        feas = re.split(' |\\t', line.strip())
        print(feas)
        if len(feas) < 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to greater than 2 dimensions" % (len(feas)))
        if not feas[0].isdigit():
             raise ValueError(
                "Unexpected label: %s, label expet to float" % (feas[0]))
        Y.append([feas[0]])
        X1 = []
        for slot in feas[1:]:
            slots = slot.split(':')
            column_name.append(slots[0])
            X1.append(slots[1])
        X.append(X1)
    df1 = pd.DataFrame(X)
    df1.columns = column_name
    df2 = pd.DataFrame(Y)
    df2.columns = ['label']
    df = pd.concat([df2, df1], axis=1)  
    return df


"""
根据特征指定的处理方案，生成对应的feature_column
feature_mapping格式类似于:
name=age;class=categorical_column_with_hash_bucket;slot_id=1;hash_bucket_size=1000
name=sex;class=categorical_column_with_vocabulary_list;slot_id=2;vocabulary_list=man,woman
name=click;class=numeric_column;slot_id=3
name=second_category;class=categorical_column_with_vocabulary_file;slot_id=4;vocabulary_file=../data/feature_map.txt;vocabulary_size=1000
name=click_lisan;class=bucketized_column;slot_id=5;source_column=click;boundries=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
name=sex_onehot;class=indicator_column;slot_id=6;categorical_column=sex
name=second_category_emb;calss=embedding_column;slot_id=7;categorical_column=second_category;dimension=100
name=age_sex;calss=crossed_columns;slot_id=8;keys=age,sex;hash_bucket_size=6000

feature_slots:
[1,2,3,4,5,6,7,8] ,feature_columns按照此格式返回
"""
def  df_to_featurecolumn(df, feature_mapping, feature_slots):
    pass




    x_train,y_train=ds.load_svmlight_file(file_name)
    print(x_train.todense(),y_train)
df = load_jimmysvm("../data/test_libfm.txt")
print(df.head())
