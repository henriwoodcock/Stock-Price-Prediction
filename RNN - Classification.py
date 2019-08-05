import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers#, Activation
KPN = pd.read_csv('/Data/Code_Data/new_data.csv',usecols=[1,2,3,4,5
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
#==============================================================================

#==============================================================================
output_shape = 1
regulariser = keras.regularizers.l2(0.01)


shape = fold1_train.shape

model = keras.Sequential()
model.add(layers.LSTM(50, input_shape = (shape[1],shape[2]), return_sequences = True,
                      #kernel_regularizer = regulariser
                      ))
#model.add(layers.Dropout(0.2))
model.add(layers.LSTM(100, return_sequences=False,
                      #kernel_regularizer = regulariser
                      ))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(output_shape, activation="sigmoid",
                       kernel_regularizer = regulariser
                       ))
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae', 'mse'])

#model.fit(X_train,y_train, epochs = 200, verbose = 1, validation_data = (X_test,y_test))
#==============================================================================

#==============================================================================
'''Training the folds:'''
model.fit(fold1_train,fold1_train_y, epochs = 100, verbose = 1)
fold1_pred = model.predict(fold1_test)

model.fit(fold2_train,fold2_train_y, epochs = 100, verbose = 1)
fold2_pred = model.predict(fold2_test)

model.fit(fold3_train,fold3_train_y, epochs = 100, verbose = 1)
fold3_pred = model.predict(fold3_test)

model.fit(fold4_train,fold4_train_y, epochs = 100, verbose = 1)
fold4_pred = model.predict(fold4_test)

model.fit(fold5_train,fold5_train_y, epochs = 100, verbose = 1)
fold5_pred = model.predict(fold5_test)
#==============================================================================

#==============================================================================
def maxi(x):
    trend = []
    for i in x:
        if i > 0.5:
            trend.append(1)
        else:
            trend.append(0)
    return trend


def find_hits(act,pred):
    '''a function to see when prediction trend is same as actual trend'''
    hits = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            hits+=1
    return hits

#==============================================================================

#==============================================================================
#fold 1:
fold1_act_trend = maxi(fold1_test_y)
fold1_pred_trend = maxi(fold1_pred)

fold1_hits = find_hits(fold1_act_trend,fold1_pred_trend)
fold1_perc = fold1_hits / len(fold1_act_trend)


#fold 2:
fold2_act_trend = maxi(fold2_test_y)
fold2_pred_trend = maxi(fold2_pred)

fold2_hits = find_hits(fold2_act_trend,fold2_pred_trend)
fold2_perc = fold2_hits / len(fold2_act_trend)


#fold 3:
fold3_act_trend = maxi(fold3_test_y)
fold3_pred_trend = maxi(fold3_pred)

fold3_hits = find_hits(fold3_act_trend,fold3_pred_trend)
fold3_perc = fold3_hits / len(fold3_act_trend)


#fold 4:
fold4_act_trend = maxi(fold4_test_y)
fold4_pred_trend = maxi(fold4_pred)

fold4_hits = find_hits(fold4_act_trend,fold4_pred_trend)
fold4_perc = fold4_hits / len(fold4_act_trend)


#fold 5:
fold5_act_trend = maxi(fold5_test_y)
fold5_pred_trend = maxi(fold5_pred)

fold5_hits = find_hits(fold5_act_trend,fold5_pred_trend)
fold5_perc = fold5_hits / len(fold5_act_trend)


def mse(act,pred):
    errr = 0
    for i in range(len(act)):
        errr += (act[i]-pred[i])**2
    errr = errr/len(act)
    return errr

def mae(act,pred):
    errr = 0
    for i in range(len(act)):
        error = act[i] - pred[i]
        if error > 0:
            errr+=error
        else:
            errr+= (-error)
    errr = errr/len(act)
    return errr



mse_5 = mse(np.array(fold5_test_y),fold5_pred)
mse_4 = mse(np.array(fold4_test_y),fold4_pred)
mse_3 = mse(np.array(fold3_test_y),fold3_pred)
mse_2 = mse(np.array(fold2_test_y),fold2_pred)
mse_1 = mse(np.array(fold1_test_y),fold1_pred)

mae_5 = mae(np.array(fold5_test_y),fold5_pred)
mae_4 = mae(np.array(fold4_test_y),fold4_pred)
mae_3 = mae(np.array(fold3_test_y),fold3_pred)
mae_2 = mae(np.array(fold2_test_y),fold2_pred)
mae_1 = mae(np.array(fold1_test_y),fold1_pred)

print(mse_5,mse_4,mse_3,mse_2,mse_1)
print(mae_5,mae_4,mae_3,mae_2,mae_1)