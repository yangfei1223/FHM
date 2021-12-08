# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
print sys.argv
print len(sys.argv)
pathname='/media/yangfei/Repository/HFM/lidar_feature/training/'
filelist=os.listdir(pathname)
filelist.sort()
# load trained model
gbm = lgb.Booster(model_file='model.txt')

for file in filelist:
    print file
    x_test = np.loadtxt(pathname+file)
    x_test = x_test[:,1:-1]
    # evaluation
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # save result
    category=file.split('_')[0]
    name=file.split('_')[1].split('.')[0]
    filename='train_pred/'+category+'_'+name+'.txt'
    np.savetxt(filename,y_pred)
