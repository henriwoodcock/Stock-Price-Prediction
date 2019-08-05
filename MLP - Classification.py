#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
reg = keras.regularizers.l2(0.01)


'''creating the model:'''
features = fold1_train.shape[1]
output_shape = 1

model = keras.Sequential()
'''layers:'''
model.add(layers.Dense(128, input_shape = (features,), activation="relu",
                       #kernel_initializer = 'uniform',
                       #kernel_regularizer = reg
                       ))
model.add(layers.Dense(128, activation = "relu",
                       #kernel_initializer = 'uniform',
                       #kernel_regularizer = reg
                       ))
model.add(layers.Dense(1, activation = 'sigmoid', kernel_regularizer = reg
                       ))
optimiser = tf.keras.optimizers.Adam()
'''compiler:'''
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
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
def maxi(x):
	'''A function used to convert the "probabilities" in classification output to a 1 or 0, 1 being an uptrend, 0 being a downtrend"'''
    trend = []
    for i in x:
        if i > 0.5:
            trend.append(1)
        else:
            trend.append(0)
    return trend

#The sets the decision rule which is if the output is greater than 0.5 (ie greater than 0.5 probability the actual output is 1 then output 1 and 0 otherwise). This is explained in my paper.
#==============================================================================

#==============================================================================
fold1_trend = maxi(fold1_predict)
fold2_trend = maxi(fold2_predict)
fold3_trend = maxi(fold3_predict)
fold4_trend = maxi(fold4_predict)
fold5_trend = maxi(fold5_predict)
#==============================================================================

#==============================================================================
def find_hits(act,pred):
    '''a function to see when prediction trend is same as actual trend'''
    hits = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            hits+=1
    return hits

fold1_hits = find_hits(fold1_trend,np.array(fold1_test_y))
fold1_perc = fold1_hits / len(fold1_trend)

fold2_hits = find_hits(fold2_trend,np.array(fold2_test_y))
fold2_perc = fold2_hits / len(fold2_trend)

fold3_hits = find_hits(fold3_trend,np.array(fold3_test_y))
fold3_perc = fold3_hits / len(fold3_trend)

fold4_hits = find_hits(fold4_trend,np.array(fold4_test_y))
fold4_perc = fold4_hits / len(fold4_trend)

fold5_hits = find_hits(fold5_trend,np.array(fold5_test_y))
fold5_perc = fold5_hits / len(fold5_trend)

print("Fold1 hits percentage", fold1_perc,
	"Fold2 hits percentage", fold2_perc,
	"Fold3 hits percentage", fold3_perc,
	"Fold4 hits percentage", fold4_perc,
	"Fold5 hits percentage", fold5_perc,)
#==============================================================================

#==============================================================================
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