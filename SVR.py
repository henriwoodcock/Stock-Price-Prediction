import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KPN = pd.read_csv('/Data/Code_Data/new_data_copy.csv', usecols=[1,2,3,4,5
                                                                ,6,7,8,9,10
                                                                ,11,12,13,14,15
                                                                ,16,17,18,19,20
                                                                ,21,22,23]
                  ).astype('float32')
#==============================================================================

#==============================================================================
'''Creating k-folds'''
sc = StandardScaler()
k = 5
set_size = len(KPN)//(k+1)
test_size = set_size//2
features = 15

#fold 1:
fold1_train = sc.fit_transform(KPN.iloc[:set_size,0:features])
#fold1_train = fold1_train.reset_index()
fold1_train_y = KPN["output"][:set_size]
fold1_test = sc.transform(KPN.iloc[set_size:2*set_size,0:features])
#fold1_test = fold1_test.reset_index()
fold1_test_y = KPN["output"][set_size:2*set_size]
fold1_test_y = fold1_test_y.reset_index(drop=True)


#fold 2:
fold2_train = KPN.iloc[:2*set_size,0:features]
#fold2_train = fold2_train.reset_index()
fold2_train_y = KPN["output"][:2*set_size]
fold2_test = KPN.iloc[2*set_size:3*set_size,0:features]
#fold2_test = fold2_test.reset_index()
fold2_test_y = KPN["output"][2*set_size:3*set_size]
fold2_test_y = fold2_test_y.reset_index(drop=True)

#fold 3:
fold3_train = KPN.iloc[:3*set_size,0:features]
#fold3_train = fold3_train.reset_index()
fold3_train_y = KPN["output"][:3*set_size]
fold3_test = KPN.iloc[3*set_size:4*set_size,0:features]
#fold3_test = fold3_test.reset_index()
fold3_test_y = KPN["output"][3*set_size:4*set_size]
fold3_test_y = fold3_test_y.reset_index(drop=True)

#fold 4:
fold4_train = KPN.iloc[:4*set_size,0:features]
#fold4_train = fold4_train.reset_index()
fold4_train_y = KPN["output"][:4*set_size]
fold4_test = KPN.iloc[4*set_size:5*set_size,0:features]
#fold4_test = fold4_test.reset_index()
fold4_test_y = KPN["output"][4*set_size:5*set_size]
fold4_test_y = fold4_test_y.reset_index(drop=True)

#fold 5:
fold5_train = KPN.iloc[:5*set_size,0:features]
#fold5_train = fold5_train.reset_index()
fold5_train_y = KPN["output"][:5*set_size]
fold5_test = KPN.iloc[5*set_size:6*set_size,0:features]
#fold5_test = fold5_test.reset_index()
fold5_test_y = KPN["output"][5*set_size:6*set_size]
fold5_test_y = fold5_test_y.reset_index(drop=True)

#==============================================================================

#==============================================================================
def SVR_Train(X,y,kern,eps,gam,c):
    piped_svr = Pipeline([('scaler', StandardScaler()),
                          ('svr', SVR(kernel=kern,epsilon=eps,gamma=gam,
                                      C = c)),
                          ])
    return piped_svr.fit(X,y)


#svr_rbf = SVR_Train(X,y,'rbf',0.01)

#y_train = svr_rbf.predict(X)


def Plot_training(actual,predicted):
    plt.plot(actual,color='black')
    plt.plot(predicted, color='red')
    return plt.show()

#Plot_training(y,y_train)

def Plot_Y_vs_Yhat(actual,predicted):
    n = len(actual)
    axis = range(0,n)
    plt.scatter(axis,actual,color='black', label='actual')
    plt.plot(axis,predicted,color='red',label='estimate')
    return plt.show()
#==============================================================================

#==============================================================================
'''
X_2 = test.iloc[:,0:6]
y_2 = test["output"]


svr_rbf = SVR_Train(X,y,'rbf',0.01)
y_rbf = svr_rbf.predict(X_2)

#Plot_Y_vs_Yhat(y_2,y_rbf)

svr_lin = SVR_Train(X,y,'linear',0.01)
y_lin = svr_lin.predict(X_2)

svr_poly = SVR_Train(X,y,'poly',0.01)
y_poly = svr_poly.predict(X_2)
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
#==============================================================================

#==============================================================================
'''
y_2 = y_2.reset_index(drop=True)

actual_trend = trend(y_2)
lin_trend = trend(y_lin)
rbf_trend = trend(y_rbf)
poly_trend = trend(y_poly)
'''
#==============================================================================

#==============================================================================
def hits(actual, pred):
    hits = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            hits +=1
    return hits
#==============================================================================

#==============================================================================
'''
lin_hits = hits(actual_trend, lin_trend)
rbf_hits = hits(actual_trend, rbf_trend)
poly_hits = hits(actual_trend, poly_trend)

lin_hit_rate = lin_hits/len(lin_trend)
rbf_hit_rate = rbf_hits/len(rbf_trend)
poly_hit_rate = poly_hits/len(poly_trend)
'''
eps = 0.01
sig_sqaured = 25
gam = 1/sig_sqaured
c = 100

'''fold 1:'''
svr_fold1 = SVR_Train(fold1_train,fold1_train_y,'rbf',eps,gam,c)
fold1_predict = svr_fold1.predict(fold1_test)

fold1_act_trend = trend(fold1_test_y)
fold1_pred_trend = trend(fold1_predict)

fold1_hits = hits(fold1_act_trend,fold1_pred_trend)
fold1_perc = fold1_hits / len(fold1_act_trend)

'''fold 2:'''
svr_fold2 = SVR_Train(fold2_train,fold2_train_y,'rbf',eps,gam,c)
fold2_predict = svr_fold2.predict(fold2_test)

fold2_act_trend = trend(fold2_test_y)
fold2_pred_trend = trend(fold2_predict)

fold2_hits = hits(fold2_act_trend,fold2_pred_trend)
fold2_perc = fold2_hits / len(fold2_act_trend)

'''fold 3:'''
svr_fold3 = SVR_Train(fold3_train,fold3_train_y,'rbf',eps,gam,c)
fold3_predict = svr_fold3.predict(fold3_test)

fold3_act_trend = trend(fold3_test_y)
fold3_pred_trend = trend(fold3_predict)

fold3_hits = hits(fold3_act_trend,fold3_pred_trend)
fold3_perc = fold3_hits / len(fold3_act_trend)

'''fold 4:'''
svr_fold4 = SVR_Train(fold4_train,fold4_train_y,'rbf',eps,gam,c)
fold4_predict = svr_fold4.predict(fold4_test)

fold4_act_trend = trend(fold4_test_y)
fold4_pred_trend = trend(fold4_predict)

fold4_hits = hits(fold4_act_trend,fold4_pred_trend)
fold4_perc = fold4_hits / len(fold4_act_trend)

'''fold 5:'''
svr_fold5 = SVR_Train(fold5_train,fold5_train_y,'rbf',eps,gam,c)
fold5_predict = svr_fold5.predict(fold5_test)

fold5_act_trend = trend(fold5_test_y)
fold5_pred_trend = trend(fold5_predict)

fold5_hits = hits(fold5_act_trend,fold5_pred_trend)
fold5_perc = fold5_hits / len(fold5_act_trend)


plt.figure()
plt.title("Support Vector Regression Fold 5 Prediction")
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