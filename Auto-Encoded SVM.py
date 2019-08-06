import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

KPN = pd.read_csv('/Data/Code_Data/new_data.csv', usecols=[1,2,3,4,5
                                                                ,6,7,8,9,10
                                                                ,11,12,13,14,15
                                                                ,16,17,18,19,20,
                                                                21,22,23]
                  ).astype('float32')
#==============================================================================

#==============================================================================
'''Creating k-folds'''
sc = StandardScaler()

k = 5
set_size = len(KPN)//(k+1)
features = 15

#fold 1:
fold1_train = sc.fit_transform(KPN.iloc[:set_size,0:features])
fold1_train_y = KPN["output"][:set_size]
fold1_test = sc.transform(KPN.iloc[set_size:2*set_size,0:features])
fold1_test_y = KPN["output"][set_size:2*set_size]
fold1_test_y = fold1_test_y.reset_index(drop=True)


#fold 2:
fold2_train = sc.fit_transform(KPN.iloc[:2*set_size,0:features])
fold2_train_y = KPN["output"][:2*set_size]
fold2_test = sc.transform(KPN.iloc[2*set_size:3*set_size,0:features])
fold2_test_y = KPN["output"][2*set_size:3*set_size]
fold2_test_y = fold2_test_y.reset_index(drop=True)

#fold 3:
fold3_train = sc.fit_transform(KPN.iloc[:3*set_size,0:features])
fold3_train_y = KPN["output"][:3*set_size]
fold3_test = sc.transform(KPN.iloc[3*set_size:4*set_size,0:features])
fold3_test_y = KPN["output"][3*set_size:4*set_size]
fold3_test_y = fold3_test_y.reset_index(drop=True)

#fold 4:
fold4_train = sc.fit_transform(KPN.iloc[:4*set_size,0:features])
fold4_train_y = KPN["output"][:4*set_size]
fold4_test = sc.transform(KPN.iloc[4*set_size:5*set_size,0:features])
fold4_test_y = KPN["output"][4*set_size:5*set_size]
fold4_test_y = fold4_test_y.reset_index(drop=True)

#fold 5:
fold5_train = sc.fit_transform(KPN.iloc[:5*set_size,0:features])
fold5_train_y = KPN["output"][:5*set_size]
fold5_test = sc.transform(KPN.iloc[5*set_size:6*set_size,0:features])
fold5_test_y = KPN["output"][5*set_size:6*set_size]
fold5_test_y = fold5_test_y.reset_index(drop=True)
#==============================================================================

#==============================================================================
'''Auto-Encoding:'''
num_neurons = 5
features = 15
regulariser = keras.regularizers.l2(0.01)

#input placeholder:
stock_in = Input(shape=(features,))
#encoded representation:
encoded = layers.Dense(num_neurons, activation="relu",
                       kernel_regularizer = regulariser
                       )(stock_in)
#decoded representation:
decoded = layers.Dense(features, activation = "linear",
                       kernel_regularizer = regulariser
                       )(encoded)

autoencoder = Model(stock_in, decoded)

encoder = Model(stock_in, encoded)
encoded_input = Input(shape=(num_neurons,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
#==============================================================================

#==============================================================================
'''auto-encoding the folds:'''
Epochs = 100
#fold 1:
autoencoder.fit(fold1_train,fold1_train, epochs = Epochs, verbose = 1)
fold1_train = encoder.predict(fold1_train)
fold1_test = encoder.predict(fold1_test)

#fold 2:
autoencoder.fit(fold2_train,fold2_train, epochs = Epochs, verbose = 1)
fold2_train = encoder.predict(fold2_train)
fold2_test = encoder.predict(fold2_test)

#fold 3:
autoencoder.fit(fold3_train,fold3_train, epochs = Epochs, verbose = 1)
fold3_train = encoder.predict(fold3_train)
fold3_test = encoder.predict(fold3_test)

#fold 4:
autoencoder.fit(fold4_train,fold4_train, epochs = Epochs, verbose = 1)
fold4_train = encoder.predict(fold4_train)
fold4_test = encoder.predict(fold4_test)

#fold 5:
autoencoder.fit(fold5_train,fold5_train, epochs = Epochs, verbose = 1)
fold5_train = encoder.predict(fold5_train)
fold5_test = encoder.predict(fold5_test)
#==============================================================================

#==============================================================================
def SVM_Train(X,y,kern,err,ga):
    piped_svm = Pipeline([#('scaler', StandardScaler()),
                        ('svm', SVC(kernel=kern, C=err, gamma = ga))
                        ])
    return piped_svm.fit(X,y)
#==============================================================================

#==============================================================================
def hits(act,pred):
    hit = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            hit +=1
    return hit

c = 78
Kernel = 'rbf'
sig_sqaured = 25
gam = 1/sig_sqaured

'''fold 1:'''
svm_fold1 = SVM_Train(fold1_train,fold1_train_y, Kernel, c, gam)
fold1_predict = svm_fold1.predict(fold1_test)

fold1_hits = hits(fold1_test_y, fold1_predict)
fold1_perc = fold1_hits/len(fold1_predict)


'''fold 2:'''
svm_fold2 = SVM_Train(fold2_train,fold2_train_y,Kernel, c, gam)
fold2_predict = svm_fold2.predict(fold2_test)

fold2_hits = hits(fold2_test_y, fold2_predict)
fold2_perc = fold2_hits/len(fold2_predict)


'''fold 3:'''
svm_fold3 = SVM_Train(fold3_train,fold3_train_y,Kernel, c, gam)
fold3_predict = svm_fold3.predict(fold3_test)

fold3_hits = hits(fold3_test_y, fold3_predict)
fold3_perc = fold3_hits/len(fold3_predict)


'''fold 4:'''
svm_fold4 = SVM_Train(fold4_train,fold4_train_y,Kernel, c, gam)
fold4_predict = svm_fold4.predict(fold4_test)

fold4_hits = hits(fold4_test_y, fold4_predict)
fold4_perc = fold4_hits/len(fold4_predict)


'''fold 5:'''
svm_fold5 = SVM_Train(fold5_train,fold5_train_y, Kernel, c, gam)
fold5_predict = svm_fold5.predict(fold5_test)

fold5_hits = hits(fold5_test_y, fold5_predict)
fold5_perc = fold5_hits/len(fold5_predict)

print("fold1 hits", fold1_perc)
print("fold2 hits", fold2_perc)
print("fold3 hits", fold3_perc)
print("fold4 hits", fold4_perc)
print("fold5 hits", fold5_perc)

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



mse_5 = mse(fold5_test_y,fold5_predict)
mse_4 = mse(fold4_test_y,fold4_predict)
mse_3 = mse(np.array(fold3_test_y),fold3_predict)
mse_2 = mse(np.array(fold2_test_y),fold2_predict)
mse_1 = mse(np.array(fold1_test_y),fold1_predict)

mae_5 = mae(np.array(fold5_test_y),fold5_predict)
mae_4 = mae(np.array(fold4_test_y),fold4_predict)
mae_3 = mae(np.array(fold3_test_y),fold3_predict)
mae_2 = mae(np.array(fold2_test_y),fold2_predict)
mae_1 = mae(np.array(fold1_test_y),fold1_predict)

