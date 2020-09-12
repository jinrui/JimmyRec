#!/usr/bin/python
# -*- coding: utf-8 -*-  

#import tensorflow as tf 
#import tensorflow.keras as ks

import numpy as np
import tensorflow as tf 
import pandas as pd 
import os
import re
import time
from sklearn.preprocessing import MinMaxScaler
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
name=age1;class=categorical_column_with_identity;slot_id=1;num_buckets=1000
name=sex;class=categorical_column_with_vocabulary_list;slot_id=2;vocabulary_list=man,woman
name=click;class=numeric_column;slot_id=3
name=second_category;class=categorical_column_with_vocabulary_file;slot_id=4;vocabulary_file=../data/feature_map.txt;vocabulary_size=1000
name=click_lisan;class=bucketized_column;slot_id=5;source_column=click;boundries=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
name=sex_onehot;class=indicator_column;slot_id=6;categorical_column=sex
name=second_category_emb;calss=embedding_column;slot_id=7;categorical_column=second_category;dimension=100
name=age_sex;calss=crossed_column;slot_id=8;keys=age,sex;hash_bucket_size=6000
name=feature_columns;class=;keys=age,sex,click,second_category,click_lisan,sex_onehot,second_category_emb,age_sex
name是特征列名字(不能重复)，class是特征处理的类，slot_id是输入的特征slot_id,second_category_emb这种slot_id会忽略。
按照feature_columns关键字对应的格式返回,
"""
def gen_featuremaps(feature_mapping):
    fea_list = []
    for line in open(feature_mapping):
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        lines = line.strip().split(';')
        tmp_map = {}
        for ll in lines:
            ls = ll.split('=')
            tmp_map[ls[0].strip()] = ls[1].strip()
        fea_list.append(tmp_map)
    return fea_list

#针对多值特征额外增加一个参数
def  make_featurecolumn(feature_conf,feature_mapping):
    fea_list = gen_featuremaps(feature_conf)
    feature_columns = []
    feature_columns_map = {}
    for fea_col in fea_list:
        fea_class = fea_col['class']
        name = fea_col['name']

        if fea_class == 'categorical_column_with_hash_bucket':
            col = tf.feature_column.categorical_column_with_hash_bucket(name, \
                hash_bucket_size = int(fea_col['hash_bucket_size']))
        if fea_class == 'numeric_column':
            col = tf.feature_column.numeric_column(name)
        if fea_class == 'categorical_column_with_identity':
            col = tf.feature_column.categorical_column_with_identity(name, \
                hash_bucket_size = int(fea_col['num_buckets']))
        if fea_class == 'categorical_column_with_vocabulary_list':
            col = tf.feature_column.categorical_column_with_vocabulary_list(name, \
                vocabulary_list = [fea.strip() for fea in fea_col['vocabulary_list'].split(',')])
        if fea_class == 'categorical_column_with_vocabulary_file':
            col = tf.feature_column.categorical_column_with_vocabulary_file(name, \
                vocabulary_file = fea_col['vocabulary_file'], vocabulary_size = int(fea_col['vocabulary_size']))
        if fea_class == 'bucketized_column':
            col = tf.feature_column.bucketized_column(feature_columns_map[fea_col['source_column']], \
                boundries = [float(fea) for fea in fea_col['boundries'].split(',')])
        if fea_class == 'indicator_column':
            col = tf.feature_column.indicator_column(categorical_column = feature_columns_map[fea_col['categorical_column']])
        if fea_class == 'embedding_column':
            col = tf.feature_column.embedding_column(categorical_column = feature_columns_map[fea_col['categorical_column']], \
                dimension = int(fea_col['dimension']))
        if fea_class == 'crossed_column':
            key_cloumns = fea_col['keys'].split(',')
            col = tf.feature_column.crossed_column([feature_columns_map[key] for key in key_cloumns], \
                hash_bucket_size = int(fea_col['hash_bucket_size']))
        feature_columns.append(col)
        feature_columns_map[name] = col
    result = []
    for line in open(feature_mapping):
        lines = line.strip().split(',')
        result.append([feature_columns_map[ll] for ll in lines])
    return result

def gen_movielens_feas(dir_name):
    #读取用户对item打分到pd
    user_item_pd = pd.read_csv(dir_name + '/u.data',sep='\t',names=['user_id', 'item_id', 'rating', 'timestamp'])
    #读取用户特征到pd
    user_feature_pd = pd.read_csv(dir_name + '/u.user',sep='\|',names=['user_id', 'age', 'gender', 'occupation','zipcode'])
    #读取item特征到pd
    item_feature_pd = pd.read_csv(dir_name + '/u.item',sep='\|',names=['item_id', 'mvtitle', 'releasedate', 'vdreleasedate','imdburl','unknow','action','adventure',
                                                                        'animation','children','comedy','crime','Documentary','Drama',
                                                                        'Fantasy','film_noir','horror','musical','mystery','romance','sci_fi','thriller','war','western'])
    user_item_pd = pd.merge(user_item_pd, user_feature_pd, on='user_id')
    user_item_pd = pd.merge(user_item_pd, item_feature_pd, on='item_id')
    def  string_toTimestamp(st):
        if type(st) != type('a') :
            return 0
        st = st.replace('Jan','1').replace('Feb','2').replace('Mar','3').replace('Apr','4').replace('May','5').replace('Jun','6').replace('Jul','7').replace('Aug','8').replace('Sep','9').replace('Oct','10').replace('Nov','11').replace('Dec','12')

        result =   time.mktime(time.strptime(st, "%d-%m-%Y"))
        return max(0, result)
    def handle_mvtitle(title):
        titles = [t.strip(' ()\'') for t in title.split(' ')]
        return titles

    user_item_pd['releasedate']=user_item_pd['releasedate'].apply(string_toTimestamp)
    user_item_pd['mvtitle']=user_item_pd['mvtitle'].apply(handle_mvtitle)
    user_item_pd[['timestamp', 'age', 'releasedate']] = MinMaxScaler().fit_transform(user_item_pd[['timestamp', 'age', 'releasedate']])
    user_item_pd['user_id'] = user_item_pd['user_id'].astype(str)
    user_item_pd['item_id'] = user_item_pd['item_id'].astype(str)
    #print(user_item_pd.head())
    #print(user_item_pd['zipcode'])
    user_item_pd = user_item_pd.drop(['vdreleasedate', 'imdburl','mvtitle','unknow'], axis=1)
    return user_item_pd