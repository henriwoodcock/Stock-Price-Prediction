import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
#==============================================================================

#==============================================================================
KPN = pd.read_csv('/Users/henriwoodcock/Documents/University/Year_3/MATH3001'
                  '/Code/Data/Code_Data/new_data_copy.csv', usecols=[1,2,3,4,5
                                                                ,6,7,8,9,10
                                                                ,11,12,13,14,15
                                                                ,16,17,18,19,20,
                                                                21,22,23]
                  ).astype('float32')

'''
train, test = Split_Train_Test(KPN, 0.1)
'''
'''training data:'''
'''
X_train = train.iloc[:,0:6] #input
y_train = train["output"] #output
'''

'''testing data:'''
'''
X_test = test.iloc[:,0:6]
y_test = test["output"]
'''

'''standardising the input data:'''
'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
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
num_neurons = 2
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

autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

'''
autoencoder.fit(X_train,X_train, epochs = 500, verbose = 1)

encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

encoded_Train = encoder.predict(X_train)
encoded_Test = encoder.predict(X_test)
'''
#==============================================================================

#==============================================================================
'''auto-encoding the folds:'''
Epochs = 100
#fold 1:
autoencoder.fit(fold1_train,fold1_train, epochs = Epochs, verbose = 1)
enc_fold1_train = encoder.predict(fold1_train)
enc_fold1_test = encoder.predict(fold1_test)

#fold 2:
autoencoder.fit(fold2_train,fold2_train, epochs = Epochs, verbose = 1)
enc_fold2_train = encoder.predict(fold2_train)
enc_fold2_test = encoder.predict(fold2_test)

#fold 3:
autoencoder.fit(fold3_train,fold3_train, epochs = Epochs, verbose = 1)
enc_fold3_train = encoder.predict(fold3_train)
enc_fold3_test = encoder.predict(fold3_test)

#fold 4:
autoencoder.fit(fold4_train,fold4_train, epochs = Epochs, verbose = 1)
enc_fold4_train = encoder.predict(fold4_train)
enc_fold4_test = encoder.predict(fold4_test)

#fold 5:
autoencoder.fit(fold5_train,fold5_train, epochs = Epochs, verbose = 1)
enc_fold5_train = encoder.predict(fold5_train)
enc_fold5_test = encoder.predict(fold5_test)
#==============================================================================

#==============================================================================
def SVR_Train(X,y,kern,eps, gam, c):
    piped_svr = Pipeline([#('scaler', StandardScaler()),
                          ('svr', SVR(kernel=kern,epsilon=eps,gamma=gam,
                                      C=c)),
                          ])
    return piped_svr.fit(X,y)
#==============================================================================

#==============================================================================
#==============================================================================
#==============================================================================
def trend(prediction):
    trend_vector = []
    for i in range(len(prediction)-1):
        if prediction[i+1] >= prediction[i]:
            trend_vector.append(1)
        else:
            trend_vector.append(-1)
    return trend_vector

def hits(actual, pred):
    hits = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            hits +=1
    return hits
#==============================================================================

#==============================================================================
eps = 0.1
sig_sqaured = 25
gam = 1/sig_sqaured
c = 100

#fold 1:
svr_fold1 = SVR_Train(enc_fold1_train,fold1_train_y,'rbf',eps,gam,c)
fold1_predict = svr_fold1.predict(enc_fold1_test)

fold1_act_trend = trend(fold1_test_y)
fold1_pred_trend = trend(fold1_predict)

fold1_hits = hits(fold1_act_trend,fold1_pred_trend)
fold1_perc = fold1_hits / len(fold1_act_trend)


#fold 2:
svr_fold2 = SVR_Train(enc_fold2_train,fold2_train_y,'rbf',eps,gam,c)
fold2_predict = svr_fold2.predict(enc_fold2_test)

fold2_act_trend = trend(fold2_test_y)
fold2_pred_trend = trend(fold2_predict)

fold2_hits = hits(fold2_act_trend,fold2_pred_trend)
fold2_perc = fold2_hits / len(fold2_act_trend)


#fold 3:
svr_fold3 = SVR_Train(enc_fold3_train,fold3_train_y,'rbf',eps,gam,c)
fold3_predict = svr_fold3.predict(enc_fold3_test)

fold3_act_trend = trend(fold3_test_y)
fold3_pred_trend = trend(fold3_predict)

fold3_hits = hits(fold3_act_trend,fold3_pred_trend)
fold3_perc = fold3_hits / len(fold3_act_trend)


#fold 4:
svr_fold4 = SVR_Train(enc_fold4_train,fold4_train_y,'rbf',eps,gam,c)
fold4_predict = svr_fold4.predict(enc_fold4_test)

fold4_act_trend = trend(fold4_test_y)
fold4_pred_trend = trend(fold4_predict)

fold4_hits = hits(fold4_act_trend,fold4_pred_trend)
fold4_perc = fold4_hits / len(fold4_act_trend)


#fold 5:
svr_fold5 = SVR_Train(enc_fold5_train,fold5_train_y,'rbf',eps,gam,c)
fold5_predict = svr_fold5.predict(enc_fold5_test)

fold5_act_trend = trend(fold5_test_y)
fold5_pred_trend = trend(fold5_predict)

fold5_hits = hits(fold5_act_trend,fold5_pred_trend)
fold5_perc = fold5_hits / len(fold5_act_trend)

plt.plot(fold5_predict)
plt.plot(fold5_test_y)



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



mse_5 = mse(np.array(fold5_test_y),fold5_predict)
mse_4 = mse(np.array(fold4_test_y),fold4_predict)
mse_3 = mse(np.array(fold3_test_y),fold3_predict)
mse_2 = mse(np.array(fold2_test_y),fold2_predict)
mse_1 = mse(np.array(fold1_test_y),fold1_predict)

mae_5 = mae(np.array(fold5_test_y),fold5_predict)
mae_4 = mae(np.array(fold4_test_y),fold4_predict)
mae_3 = mae(np.array(fold3_test_y),fold3_predict)
mae_2 = mae(np.array(fold2_test_y),fold2_predict)
mae_1 = mae(np.array(fold1_test_y),fold1_predict)