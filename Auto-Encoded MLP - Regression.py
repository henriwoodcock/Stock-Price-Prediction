import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model



#Import the data:
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
num_neurons = 3
regulariser = keras.regularizers.l2(0.01)

#input placeholder:
stock_in = Input(shape=(features,))
#encoded representation:
encoded = layers.Dense(num_neurons, activation="relu",
                       #kernel_regularizer = regulariser
                       )(stock_in)
#decoded representation:
decoded = layers.Dense(features, activation = "linear",
                       #kernel_regularizer = regulariser
                       )(encoded)
autoencoder = Model(stock_in, decoded)
encoder = Model(stock_in, encoded)
encoded_input = Input(shape=(num_neurons,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
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

EPOCHS = 200
regulariser = keras.regularizers.l2(0.01)

'''creating the model:'''
features = enc_fold1_train.shape[1]
output_shape = 1

model = keras.Sequential()
'''layers:'''
model.add(layers.Dense(128, input_shape = (features,), activation="relu",
                       kernel_regularizer = regulariser
                       ))
model.add(layers.Dense(128, activation = "relu",
                       kernel_regularizer = regulariser
                       ))
model.add(layers.Dense(1,
                       kernel_regularizer = regulariser,
                       activation="linear"
                       ))
#optimiser = tf.keras.optimizers.Adam()
'''compiler:'''
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mse'])

#==============================================================================

#==============================================================================
#fold 1:
model.fit(enc_fold1_train, fold1_train_y, epochs = EPOCHS, verbose = 1)
fold1_predict = model.predict(enc_fold1_test)

#fold 2:
model.fit(enc_fold2_train, fold2_train_y, epochs = EPOCHS, verbose = 1)
fold2_predict = model.predict(enc_fold2_test)

#fold 3:
model.fit(enc_fold3_train, fold3_train_y, epochs = EPOCHS, verbose = 1)
fold3_predict = model.predict(enc_fold3_test)

#fold 4:
model.fit(enc_fold4_train, fold4_train_y, epochs = EPOCHS, verbose = 1)
fold4_predict = model.predict(enc_fold4_test)

#fold 5:
model.fit(enc_fold5_train, fold5_train_y, epochs = EPOCHS, verbose = 1)
fold5_predict = model.predict(enc_fold5_test)
#==============================================================================

#==============================================================================

#Compare movements:
def find_trend(prices):
    '''function to create a list of the price trend'''
    trend = []
    for i in range(len(prices)-1):
        if prices[i+1] > prices[i]:
            trend.append(1)
        else:
            trend.append(-1)
    return trend

fold1_trend = find_trend(np.array(fold2_test_y)) #1 for up, -1 for down
fold1_pred_trend = find_trend(fold2_predict)

fold2_trend = find_trend(np.array(fold2_test_y)) #1 for up, -1 for down
fold2_pred_trend = find_trend(fold2_predict)

fold3_trend = find_trend(np.array(fold3_test_y)) #1 for up, -1 for down
fold3_pred_trend = find_trend(fold3_predict)

fold4_trend = find_trend(np.array(fold4_test_y)) #1 for up, -1 for down
fold4_pred_trend = find_trend(fold4_predict)

fold5_trend = find_trend(np.array(fold5_test_y)) #1 for up, -1 for down
fold5_pred_trend = find_trend(fold5_predict)


def find_hits(act,pred):
    '''a function to see when prediction trend is same as actual trend'''
    hits = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            hits+=1
    return hits

fold1_hits = find_hits(fold1_trend,fold1_pred_trend)
fold1_perc = fold1_hits / len(fold1_trend)

fold2_hits = find_hits(fold2_trend,fold2_pred_trend)
fold2_perc = fold2_hits / len(fold2_trend)

fold3_hits = find_hits(fold3_trend,fold3_pred_trend)
fold3_perc = fold3_hits / len(fold3_trend)

fold4_hits = find_hits(fold4_trend,fold4_pred_trend)
fold4_perc = fold4_hits / len(fold4_trend)

fold5_hits = find_hits(fold5_trend,fold5_pred_trend)
fold5_perc = fold5_hits / len(fold5_trend)

plt.plot(np.array(fold5_test_y))
plt.plot(fold5_predict)

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

print(mse_1,mse_2,mse_3,mse_4,mse_5)
print(mae_1,mae_2,mae_3,mae_4,mae_5)