# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
print sys.argv
print len(sys.argv)
pathname='/home/yangfei/Datasets/lidar_feature/training/'
# load data
print('Load data...')
filelist = os.listdir(pathname)
filelist.sort()
data_li=[]
for filename in filelist:
    print filename
    mat = np.loadtxt(pathname+filename)
    data_li.append(mat)

train_data=np.vstack(data_li)
# train data
# w_train=train_data[0].values
y_train=train_data[:,0]
x_train=train_data[:,1:]


# create dataset
lgb_train=lgb.Dataset(x_train,y_train)

# para dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq':1,
    'num_leaves': 31,
    'tree_learner':'serial',
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# train
print('Start training...')
gbm=lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_train,early_stopping_rounds=10)

# save model
print('Save model...')
gbm.save_model('model.txt')


print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))


