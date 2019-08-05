import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Import the data:
KPN = pd.read_csv('/Data/Code_Data/new_data_copy.csv', usecols=[1,2,3,4,5
                                                                ,6,7,8,9,10
                                                                ,11,12,13,14,15
                                                                ,16,17,18,19,20,
                                                                21,22,23]
                  ).astype('float32')

def Split_Train_Test(data, test_ratio):
    '''splits data into a training and testing set'''
    train_set_size = 1 - int(len(data) * test_ratio)
    train_set = data[:train_set_size]
    test_set = data[train_set_size:]
    return train_set, test_set
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

#fold 2:
fold2_train = sc.fit_transform(KPN.iloc[:2*set_size,0:features])
fold2_train_y = KPN["output"][:2*set_size]
fold2_test = sc.transform(KPN.iloc[2*set_size:3*set_size,0:features])
fold2_test_y = KPN["output"][2*set_size:3*set_size]

#fold 3:
fold3_train = sc.fit_transform(KPN.iloc[:3*set_size,0:features])
fold3_train_y = KPN["output"][:3*set_size]
fold3_test = sc.transform(KPN.iloc[3*set_size:4*set_size,0:features])
fold3_test_y = KPN["output"][3*set_size:4*set_size]

#fold 4:
fold4_train = sc.fit_transform(KPN.iloc[:4*set_size,0:features])
fold4_train_y = KPN["output"][:4*set_size]
fold4_test = sc.transform(KPN.iloc[4*set_size:5*set_size,0:features])
fold4_test_y = KPN["output"][4*set_size:5*set_size]

#fold 5:
fold5_train = sc.fit_transform(KPN.iloc[:5*set_size,0:features])
fold5_train_y = KPN["output"][:5*set_size]
fold5_test = sc.transform(KPN.iloc[5*set_size:6*set_size,0:features])
fold5_test_y = KPN["output"][5*set_size:6*set_size]
#==============================================================================

#==============================================================================
EPOCHS = 200
regulariser = keras.regularizers.l2(0.01)

'''creating the model:'''
features = fold1_train.shape[1]
output_shape = 1

model = keras.Sequential()
'''layers:'''
model.add(layers.Dense(128, input_shape = (features,), activation="relu",
                       #kernel_regularizer = regulariser
                       ))
model.add(layers.Dense(128, activation = "relu",
                       #kernel_regularizer = regulariser
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
model.fit(fold1_train, fold1_train_y, epochs = EPOCHS, verbose = 1)
fold1_predict = model.predict(fold1_test)

#fold 2:
model.fit(fold2_train, fold2_train_y, epochs = EPOCHS, verbose = 1)
fold2_predict = model.predict(fold2_test)

#fold 3:
model.fit(fold3_train, fold3_train_y, epochs = EPOCHS, verbose = 1)
fold3_predict = model.predict(fold3_test)

#fold 4:
model.fit(fold4_train, fold4_train_y, epochs = EPOCHS, verbose = 1)
fold4_predict = model.predict(fold4_test)

#fold 5:
model.fit(fold5_train, fold5_train_y, epochs = EPOCHS, verbose = 1)
fold5_predict = model.predict(fold5_test)
#==============================================================================

#==============================================================================
'''viewing model improvements'''
'''
history = model.fit(fold5_train,fold5_train_y, epochs = EPOCHS, verbose = 1,
                    validation_data = (fold5_test, fold5_test_y))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])

plot_history(history)
'''
#==============================================================================

#==============================================================================
'''predictions:'''
'''
#On training data:
y_train_hat = model.predict(X_train)
plt.figure('Training Predictions')
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.plot(y_train, label="Actual prices")
plt.plot(y_train_hat, label="Predicted prices")
plt.legend()
plt.show()

#On test data:
y_test_2 = y_test.reset_index(drop=True)
y_test_hat = model.predict(X_test)
plt.figure('Testing Predictions')
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.plot(y_test_2, label = "actual")
plt.plot(y_test_hat, label = "predictions")
plt.legend()
plt.show()
'''
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

plt.figure()
plt.title("Feed Forward ANN Fold 5 Prediction")
plt.plot(np.array(fold5_test_y), label="Actual")
plt.plot(fold5_predict, label="Predicted")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")


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