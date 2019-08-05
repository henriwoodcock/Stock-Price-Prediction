import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers#, Activation
KPN = pd.read_csv('/Users/henriwoodcock/Documents/University/Year_3/MATH3001'
                  '/Code/Data/Code_Data/new_data.csv',usecols=[1,2,3,4,5
                                                                ,6,7,8,9,10
                                                                ,11,12,13,14,15
                                                                ,16,17,18,19,20,
                                                                21,22,23]
                  ).astype('float32')

def Split_Train_Test(data, test_ratio):
    test_set_size = int(len(data) * test_ratio)
    test_set = data[:test_set_size]
    train_set = data[test_set_size:]
    return train_set, test_set
#==============================================================================

#==============================================================================
    '''Creating k-folds'''
sc = StandardScaler()

'''Slightly different structure when creating the K folds for RNN, because RNN is designed for pictures it is usually sent a 3d data source for RGB so here we use 1st dimension for input, 2nd dimension for amount of time steps, here is 1 per input/output and 3rd dimension for output'''

k = 5
set_size = len(KPN)//(k+1)
features = 15

#fold 1:
fold1_train = np.array(sc.fit_transform(KPN.iloc[:set_size,0:features]))
fold1_train_y = KPN["output"][:set_size]
fold1_test = np.array(sc.transform(KPN.iloc[set_size:2*set_size,0:features]))
fold1_test_y = KPN["output"][set_size:2*set_size]
fold1_test_y = fold1_test_y.reset_index(drop=True)

fold1_train = np.reshape(fold1_train,(fold1_train.shape[0], 1,
                                      fold1_train.shape[1]))
fold1_test = np.reshape(fold1_test,(fold1_test.shape[0], 1,
                                      fold1_test.shape[1]))


#fold 2:
fold2_train = np.array(sc.fit_transform(KPN.iloc[:2*set_size,0:features]))
fold2_train_y = KPN["output"][:2*set_size]
fold2_test = np.array(sc.transform(KPN.iloc[2*set_size:3*set_size,0:features]))
fold2_test_y = KPN["output"][2*set_size:3*set_size]
fold2_test_y = fold2_test_y.reset_index(drop=True)

fold2_train = np.reshape(fold2_train,(fold2_train.shape[0], 1,
                                      fold2_train.shape[1]))
fold2_test = np.reshape(fold2_test,(fold2_test.shape[0], 1,
                                      fold2_test.shape[1]))


#fold 3:
fold3_train = np.array(sc.fit_transform(KPN.iloc[:3*set_size,0:features]))
fold3_train_y = KPN["output"][:3*set_size]
fold3_test = np.array(sc.transform(KPN.iloc[3*set_size:4*set_size,0:features]))
fold3_test_y = KPN["output"][3*set_size:4*set_size]
fold3_test_y = fold3_test_y.reset_index(drop=True)

fold3_train = np.reshape(fold3_train,(fold3_train.shape[0], 1,
                                      fold3_train.shape[1]))
fold3_test = np.reshape(fold3_test,(fold3_test.shape[0], 1,
                                      fold3_test.shape[1]))


#fold 4:
fold4_train = np.array(sc.fit_transform(KPN.iloc[:4*set_size,0:features]))
fold4_train_y = KPN["output"][:4*set_size]
fold4_test = np.array(sc.transform(KPN.iloc[4*set_size:5*set_size,0:features]))
fold4_test_y = KPN["output"][4*set_size:5*set_size]
fold4_test_y = fold4_test_y.reset_index(drop=True)

fold4_train = np.reshape(fold4_train,(fold4_train.shape[0], 1,
                                      fold4_train.shape[1]))
fold4_test = np.reshape(fold4_test,(fold4_test.shape[0], 1,
                                      fold4_test.shape[1]))


#fold 5:
fold5_train = np.array(sc.fit_transform(KPN.iloc[:5*set_size,0:features]))
fold5_train_y = KPN["output"][:5*set_size]
fold5_test = np.array(sc.transform(KPN.iloc[5*set_size:6*set_size,0:features]))
fold5_test_y = KPN["output"][5*set_size:6*set_size]
fold5_test_y = fold5_test_y.reset_index(drop=True)

fold5_train = np.reshape(fold5_train,(fold5_train.shape[0], 1,
                                      fold5_train.shape[1]))
fold5_test = np.reshape(fold5_test,(fold5_test.shape[0], 1,
                                      fold5_test.shape[1]))