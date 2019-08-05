import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model



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
'''
train, test = Split_Train_Test(KPN, 0.1)

''''''training data:''''''
X_train = train.iloc[:,0:6] #input
y_train = train["output"] #output

''''''testing data:''''''
X_test = test.iloc[:,0:6]
y_test = test["output"]

''''''standardising the input data:''''''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.reshape(X_train,(X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0], 1, X_test.shape[1]))
'''
#==============================================================================

#==============================================================================
'''Creating k-folds'''
sc = StandardScaler()


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


model = keras.Sequential()
model.add(layers.LSTM(100, input_shape = (1,features), return_sequences = True,
                      #kernel_regularizer = regulariser
                      ))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(100, return_sequences=False,
                      #kernel_regularizer = regulariser
                      ))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(output_shape, activation="linear",
                       kernel_regularizer = regulariser
                       ))
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae', 'mse'])

#model.fit(X_train,y_train, epochs = 100, verbose = 1)
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
#fold 1:
fold1_act_trend = trend(fold1_test_y)
fold1_pred_trend = trend(fold1_pred)

fold1_hits = hits(fold1_act_trend,fold1_pred_trend)
fold1_perc = fold1_hits / len(fold1_act_trend)


#fold 2:
fold2_act_trend = trend(fold2_test_y)
fold2_pred_trend = trend(fold2_pred)

fold2_hits = hits(fold2_act_trend,fold2_pred_trend)
fold2_perc = fold2_hits / len(fold2_act_trend)


#fold 3:
fold3_act_trend = trend(fold3_test_y)
fold3_pred_trend = trend(fold3_pred)

fold3_hits = hits(fold3_act_trend,fold3_pred_trend)
fold3_perc = fold3_hits / len(fold3_act_trend)


#fold 4:
fold4_act_trend = trend(fold4_test_y)
fold4_pred_trend = trend(fold4_pred)

fold4_hits = hits(fold4_act_trend,fold4_pred_trend)
fold4_perc = fold4_hits / len(fold4_act_trend)


#fold 5:
fold5_act_trend = trend(fold5_test_y)
fold5_pred_trend = trend(fold5_pred)

fold5_hits = hits(fold5_act_trend,fold5_pred_trend)
fold5_perc = fold5_hits / len(fold5_act_trend)

plt.figure()
plt.title("RNN Regression Fold 5 Prediction")
plt.plot(np.array(fold5_test_y), label="Actual")
plt.plot(fold5_pred, label="Predicted")
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

print(mse_1,mse_2,mse_3,mse_4,mse_5)
print(mae_1,mae_2,mae_3,mae_4,mae_5)