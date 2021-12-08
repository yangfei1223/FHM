# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse

print sys.argv
print len(sys.argv)
pathname='/home/yangfei/Datasets/Result/KITTI/'+sys.argv[1]+'/Lidar/'
# load data
print('Load data...')
train_data=pd.read_csv(pathname+'denseFeature.train',header=None,sep='\t')
# train data
w_train=train_data[0].values
y_train=train_data[1].values
x_train=train_data.drop([0,1],axis=1).values

# create dataset
lgb_train=lgb.Dataset(x_train,y_train,weight=w_train)

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
gbm.save_model(pathname+'dense_model_'+sys.argv[1]+'.txt')

'''
# dump model to json (and save to file)
print('Dump model to JSON...')
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

'''
print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))


