#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

KPN = pd.read_csv('/Data/Code_Data/new_data.csv', usecols=[1,2,3,4,5,
                                                                6,7,8,9,10,
                                                                11,12,13,14,
                                                                15,16,17,18,
                                                                19,20,21,22,23]
                  ).astype('float32')
#==============================================================================

#==============================================================================
'''Creating k-folds'''

k = 5
set_size = len(KPN)//(k+1)
features = 15

#fold 1:
fold1_train = KPN.iloc[:set_size,0:features]
fold1_train_y = KPN["output"][:set_size]
fold1_test = KPN.iloc[set_size:2*set_size,0:features]
fold1_test_y = KPN["output"][set_size:2*set_size]
fold1_test_y = fold1_test_y.reset_index(drop=True)


#fold 2:
fold2_train = KPN.iloc[:2*set_size,0:features]
fold2_train_y = KPN["output"][:2*set_size]
fold2_test = KPN.iloc[2*set_size:3*set_size,0:features]
fold2_test_y = KPN["output"][2*set_size:3*set_size]
fold2_test_y = fold2_test_y.reset_index(drop=True)

#fold 3:
fold3_train = KPN.iloc[:3*set_size,0:features]
fold3_train_y = KPN["output"][:3*set_size]
fold3_test = KPN.iloc[3*set_size:4*set_size,0:features]
fold3_test_y = KPN["output"][3*set_size:4*set_size]
fold3_test_y = fold3_test_y.reset_index(drop=True)

#fold 4:
fold4_train = KPN.iloc[:4*set_size,0:features]
fold4_train_y = KPN["output"][:4*set_size]
fold4_test = KPN.iloc[4*set_size:5*set_size,0:features]
fold4_test_y = KPN["output"][4*set_size:5*set_size]
fold4_test_y = fold4_test_y.reset_index(drop=True)

#fold 5:
fold5_train = KPN.iloc[:5*set_size,0:features]
fold5_train_y = KPN["output"][:5*set_size]
fold5_test = KPN.iloc[5*set_size:6*set_size,0:features]
fold5_test_y = KPN["output"][5*set_size:6*set_size]
fold5_test_y = fold5_test_y.reset_index(drop=True)
#==============================================================================

#==============================================================================
def SVM_Train(X,y,kern,err,ga):
	'''A function to fit a SVM model'''
    piped_svm = Pipeline([('scaler', StandardScaler()),
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

'''Trial and error method used here to find best parameter values ^^^^, however this could be updated to a grid search method for more efficiency and to remove human error'''

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

#Results
print("fold1 hit percentage", fold1_perc)
print("fold2 hits percentage", fold2_perc)
print("fold3 hits percentage", fold3_perc)
print("fold4 hits percentage", fold4_perc)
print("fold5 hits percentage", fold5_perc)

'''Function and statistics for Comparisons'''
def mse(act,pred):
    error = 0
    for i in range(len(act)):
        error += (act[i]-pred[i])**2
    error = error/len(act)
    return error

def mae(act,pred):
    tot_error = 0
    for i in range(len(act)):
        error = act[i] - pred[i]
        if error > 0:
            tot_error+=error
        else:
            tot_error+= (-error)
    tot_error = tot_error/len(act)
    return tot_error



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