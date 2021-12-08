# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
print sys.argv
print len(sys.argv)
pathname='/home/yangfei/Datasets/Result/KITTI/val/Lidar/'
# load trained model
gbm = lgb.Booster(model_file=pathname+'dense_model_val.txt')

for i in range(1,int(sys.argv[2]),2):
    filename=pathname+'denseFeatureEval/'+sys.argv[1]+'_%06d.test' %(i)
    print filename
    test_data = pd.read_csv(filename, header=None, sep='\t')
    # y_test = test_data[0].values
    x_test = test_data.values
    # evaluation
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # save result
    filename=pathname+'denseClassifierOut/'+sys.argv[1]+'_%06d.txt' %(i)
    np.savetxt(filename,y_pred)
