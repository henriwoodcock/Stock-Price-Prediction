import numpy as np
import pandas as pd

MAE = pd.read_csv('/Data/MAE.csv',index_col=0).astype('float32')
MSE = pd.read_csv('/Data/MSE.csv',index_col=0).astype('float32')
HITS = pd.read_csv('/Data/HITS.csv',index_col=0).astype('float32')

Fold1_Hits = HITS["Fold 1"]
Fold2_Hits = HITS["Fold 2"]
Fold3_Hits = HITS["Fold 3"]
Fold4_Hits = HITS["Fold 4"]
Fold5_Hits = HITS["Fold 5"]

Fold1_MSE = MSE["Fold 1"]
Fold2_MSE = MSE["Fold 2"]
Fold3_MSE = MSE["Fold 3"]
Fold4_MSE = MSE["Fold 4"]
Fold5_MSE = MSE["Fold 5"]

Fold1_MAE = MAE["Fold 1"]
Fold2_MAE = MAE["Fold 2"]
Fold3_MAE = MAE["Fold 3"]
Fold4_MAE = MAE["Fold 4"]
Fold5_MAE = MAE["Fold 5"]

def avg(vec):
    summ = 0
    for i in vec:
        summ+=i
    return summ/len(vec)


def make_class(y):
    return [y["SVM"],y["MLP"],y["RNN"], y["AE SVM"], y["AE MLP"], y["AE RNN"]]
def make_class_noAE(y):
    return [y["SVM"],y["MLP"],y["RNN"]]
def make_class_AE(y):
    return [y["AE SVM"], y["AE MLP"], y["AE RNN"]]


def make_reg(y):
    return [y["SVR"],y["MLP Reg"],y["RNN Reg"], y["AE SVR"],
            y["AE MLP Reg"], y["AE RNN Reg"]]
def make_reg_noAE(y):
    return [y["SVR"],y["MLP Reg"],y["RNN Reg"]]
def make_reg_AE(y):
    return [y["AE SVR"], y["AE MLP Reg"], y["AE RNN Reg"]]


'''Classification_Avg_Hits'''
Class_Fold1_hits = make_class(Fold1_Hits)
Class_Fold1_avg = avg(Class_Fold1_hits)

Class_Fold2_hits = make_class(Fold2_Hits)
Class_Fold2_avg = avg(Class_Fold2_hits)

Class_Fold3_hits = make_class(Fold3_Hits)
Class_Fold3_avg = avg(Class_Fold3_hits)

Class_Fold4_hits = make_class(Fold4_Hits)
Class_Fold4_avg = avg(Class_Fold4_hits)

Class_Fold5_hits = make_class(Fold5_Hits)
Class_Fold5_avg = avg(Class_Fold5_hits)

'''Classification AE AVG Hits'''
Class_Fold1_AEhits = make_class_AE(Fold1_Hits)
Class_Fold1_AEavg = avg(Class_Fold1_AEhits)

Class_Fold2_AEhits = make_class_AE(Fold2_Hits)
Class_Fold2_AEavg = avg(Class_Fold2_AEhits)

Class_Fold3_AEhits = make_class_AE(Fold3_Hits)
Class_Fold3_AEavg = avg(Class_Fold3_AEhits)

Class_Fold4_AEhits = make_class_AE(Fold4_Hits)
Class_Fold4_AEavg = avg(Class_Fold4_AEhits)

Class_Fold5_AEhits = make_class_AE(Fold5_Hits)
Class_Fold5_AEavg = avg(Class_Fold5_AEhits)

'''Classification No AE AVG Hits'''
Class_Fold1_noAEhits = make_class_noAE(Fold1_Hits)
Class_Fold1_noAEavg = avg(Class_Fold1_noAEhits)

Class_Fold2_noAEhits = make_class_noAE(Fold2_Hits)
Class_Fold2_noAEavg = avg(Class_Fold2_noAEhits)

Class_Fold3_noAEhits = make_class_noAE(Fold3_Hits)
Class_Fold3_noAEavg = avg(Class_Fold3_noAEhits)

Class_Fold4_noAEhits = make_class_noAE(Fold4_Hits)
Class_Fold4_noAEavg = avg(Class_Fold4_noAEhits)

Class_Fold5_noAEhits = make_class_noAE(Fold5_Hits)
Class_Fold5_noAEavg = avg(Class_Fold5_noAEhits)





'''Regression_AVG_hits'''
Reg_Fold1_hits = make_reg(Fold1_Hits)
Reg_Fold1_avg = avg(Reg_Fold1_hits)

Reg_Fold2_hits = make_reg(Fold2_Hits)
Reg_Fold2_avg = avg(Reg_Fold2_hits)

Reg_Fold3_hits = make_reg(Fold3_Hits)
Reg_Fold3_avg = avg(Reg_Fold3_hits)

Reg_Fold4_hits = make_reg(Fold4_Hits)
Reg_Fold4_avg = avg(Reg_Fold4_hits)

Reg_Fold5_hits = make_reg(Fold5_Hits)
Reg_Fold5_avg = avg(Reg_Fold5_hits)

'''Regression Hits AE'''
Reg_Fold1_AEhits = make_reg_AE(Fold1_Hits)
Reg_Fold1_AEavg = avg(Reg_Fold1_AEhits)

Reg_Fold2_AEhits = make_reg_AE(Fold2_Hits)
Reg_Fold2_AEavg = avg(Reg_Fold2_AEhits)

Reg_Fold3_AEhits = make_reg_AE(Fold3_Hits)
Reg_Fold3_AEavg = avg(Reg_Fold3_AEhits)

Reg_Fold4_AEhits = make_reg_AE(Fold4_Hits)
Reg_Fold4_AEavg = avg(Reg_Fold4_AEhits)

Reg_Fold5_AEhits = make_reg_AE(Fold5_Hits)
Reg_Fold5_AEavg = avg(Reg_Fold5_AEhits)

'''Regression Hits No AE'''
Reg_Fold1_noAEhits = make_reg_noAE(Fold1_Hits)
Reg_Fold1_noAEavg = avg(Reg_Fold1_noAEhits)

Reg_Fold2_noAEhits = make_reg_noAE(Fold2_Hits)
Reg_Fold2_noAEavg = avg(Reg_Fold2_noAEhits)

Reg_Fold3_noAEhits = make_reg_noAE(Fold3_Hits)
Reg_Fold3_noAEavg = avg(Reg_Fold3_noAEhits)

Reg_Fold4_noAEhits = make_reg_noAE(Fold4_Hits)
Reg_Fold4_noAEavg = avg(Reg_Fold4_noAEhits)

Reg_Fold5_noAEhits = make_reg_noAE(Fold5_Hits)
Reg_Fold5_noAEavg = avg(Reg_Fold5_noAEhits)





'''Classification MAE'''
Class_Fold1_MAE = make_class(Fold1_MAE)
Class_Fold1_avgMAE = avg(Class_Fold1_MAE)

Class_Fold2_MAE = make_class(Fold2_MAE)
Class_Fold2_avgMAE = avg(Class_Fold2_MAE)

Class_Fold3_MAE = make_class(Fold3_MAE)
Class_Fold3_avgMAE = avg(Class_Fold3_MAE)

Class_Fold4_MAE = make_class(Fold4_MAE)
Class_Fold4_avgMAE = avg(Class_Fold4_MAE)

Class_Fold5_MAE = make_class(Fold5_MAE)
Class_Fold5_avgMAE = avg(Class_Fold5_MAE)

'''Classification AE MAE'''
Class_Fold1_AEMAE = make_class_AE(Fold1_MAE)
Class_Fold1_AEavgMAE = avg(Class_Fold1_AEMAE)

Class_Fold2_AEMAE = make_class_AE(Fold2_MAE)
Class_Fold2_AEavgMAE = avg(Class_Fold2_AEMAE)

Class_Fold3_AEMAE = make_class_AE(Fold3_MAE)
Class_Fold3_AEavgMAE = avg(Class_Fold3_AEMAE)

Class_Fold4_AEMAE = make_class_AE(Fold4_MAE)
Class_Fold4_AEavgMAE = avg(Class_Fold4_AEMAE)

Class_Fold5_AEMAE = make_class_AE(Fold5_MAE)
Class_Fold5_AEavgMAE = avg(Class_Fold5_AEMAE)

'''Classification No AE MAE'''
Class_Fold1_noAEMAE = make_class_noAE(Fold1_MAE)
Class_Fold1_noAEavgMAE = avg(Class_Fold1_noAEMAE)

Class_Fold2_noAEMAE = make_class_noAE(Fold2_MAE)
Class_Fold2_noAEavgMAE = avg(Class_Fold2_noAEMAE)

Class_Fold3_noAEMAE = make_class_noAE(Fold3_MAE)
Class_Fold3_noAEavgMAE = avg(Class_Fold3_noAEMAE)

Class_Fold4_noAEMAE = make_class_noAE(Fold4_MAE)
Class_Fold4_noAEavgMAE = avg(Class_Fold4_noAEMAE)

Class_Fold5_noAEMAE = make_class_noAE(Fold5_MAE)
Class_Fold5_noAEavgMAE = avg(Class_Fold5_noAEMAE)





'''Regression MAE'''
Reg_Fold1_MAE = make_reg(Fold1_MAE)
Reg_Fold1_avgMAE = avg(Reg_Fold1_MAE)

Reg_Fold2_MAE = make_reg(Fold2_MAE)
Reg_Fold2_avgMAE = avg(Reg_Fold2_MAE)

Reg_Fold3_MAE = make_reg(Fold3_MAE)
Reg_Fold3_avgMAE = avg(Reg_Fold3_MAE)

Reg_Fold4_MAE = make_reg(Fold4_MAE)
Reg_Fold4_avgMAE = avg(Reg_Fold4_MAE)

Reg_Fold5_MAE = make_reg(Fold5_MAE)
Reg_Fold5_avgMAE = avg(Reg_Fold5_MAE)

'''Regression MAE AE'''
Reg_Fold1_AEMAE = make_reg_AE(Fold1_MAE)
Reg_Fold1_AEavgMAE = avg(Reg_Fold1_AEMAE)

Reg_Fold2_AEMAE = make_reg_AE(Fold2_MAE)
Reg_Fold2_AEavgMAE = avg(Reg_Fold2_AEMAE)

Reg_Fold3_AEMAE = make_reg_AE(Fold3_MAE)
Reg_Fold3_AEavgMAE = avg(Reg_Fold3_AEMAE)

Reg_Fold4_AEMAE = make_reg_AE(Fold4_MAE)
Reg_Fold4_AEavgMAE = avg(Reg_Fold4_AEMAE)

Reg_Fold5_AEMAE = make_reg_AE(Fold5_MAE)
Reg_Fold5_AEavgMAE = avg(Reg_Fold5_AEMAE)

'''Regression MAE No AE'''
Reg_Fold1_noAEMAE = make_reg_noAE(Fold1_MAE)
Reg_Fold1_noAEavgMAE = avg(Reg_Fold1_noAEMAE)

Reg_Fold2_noAEMAE = make_reg_noAE(Fold2_MAE)
Reg_Fold2_noAEavgMAE = avg(Reg_Fold2_noAEMAE)

Reg_Fold3_noAEMAE = make_reg_noAE(Fold3_MAE)
Reg_Fold3_noAEavgMAE = avg(Reg_Fold3_noAEMAE)

Reg_Fold4_noAEMAE = make_reg_noAE(Fold4_MAE)
Reg_Fold4_noAEavgMAE = avg(Reg_Fold4_noAEMAE)

Reg_Fold5_noAEMAE = make_reg_noAE(Fold5_MAE)
Reg_Fold5_noAEavgMAE = avg(Reg_Fold5_noAEMAE)





'''Classification MSE'''
Class_Fold1_MSE = make_class(Fold1_MSE)
Class_Fold1_avgMSE = avg(Class_Fold1_MSE)

Class_Fold2_MSE = make_class(Fold2_MSE)
Class_Fold2_avgMSE = avg(Class_Fold2_MSE)

Class_Fold3_MSE = make_class(Fold3_MSE)
Class_Fold3_avgMSE = avg(Class_Fold3_MSE)

Class_Fold4_MSE = make_class(Fold4_MSE)
Class_Fold4_avgMSE = avg(Class_Fold4_MSE)

Class_Fold5_MSE = make_class(Fold5_MSE)
Class_Fold5_avgMSE = avg(Class_Fold5_MSE)

'''Classification AE MSE'''
Class_Fold1_AEMSE = make_class_AE(Fold1_MSE)
Class_Fold1_AEavgMSE = avg(Class_Fold1_AEMSE)

Class_Fold2_AEMSE = make_class_AE(Fold2_MSE)
Class_Fold2_AEavgMSE = avg(Class_Fold2_AEMSE)

Class_Fold3_AEMSE = make_class_AE(Fold3_MSE)
Class_Fold3_AEavgMSE = avg(Class_Fold3_AEMSE)

Class_Fold4_AEMSE = make_class_AE(Fold4_MSE)
Class_Fold4_AEavgMSE = avg(Class_Fold4_AEMSE)

Class_Fold5_AEMSE = make_class_AE(Fold5_MSE)
Class_Fold5_AEavgMSE = avg(Class_Fold5_AEMSE)

'''Classification No AE MSE'''
Class_Fold1_noAEMSE = make_class_noAE(Fold1_MSE)
Class_Fold1_noAEavgMSE = avg(Class_Fold1_noAEMSE)

Class_Fold2_noAEMSE = make_class_noAE(Fold2_MSE)
Class_Fold2_noAEavgMSE = avg(Class_Fold2_noAEMSE)

Class_Fold3_noAEMSE = make_class_noAE(Fold3_MSE)
Class_Fold3_noAEavgMSE = avg(Class_Fold3_noAEMSE)

Class_Fold4_noAEMSE = make_class_noAE(Fold4_MSE)
Class_Fold4_noAEavgMSE = avg(Class_Fold4_noAEMSE)

Class_Fold5_noAEMSE = make_class_noAE(Fold5_MSE)
Class_Fold5_noAEavgMSE = avg(Class_Fold5_noAEMSE)





'''Regression MSE'''
Reg_Fold1_MSE = make_reg(Fold1_MSE)
Reg_Fold1_avgMSE = avg(Reg_Fold1_MSE)

Reg_Fold2_MSE = make_reg(Fold2_MSE)
Reg_Fold2_avgMSE = avg(Reg_Fold2_MSE)

Reg_Fold3_MSE = make_reg(Fold3_MSE)
Reg_Fold3_avgMSE = avg(Reg_Fold3_MSE)

Reg_Fold4_MSE = make_reg(Fold4_MSE)
Reg_Fold4_avgMSE = avg(Reg_Fold4_MSE)

Reg_Fold5_MSE = make_reg(Fold5_MSE)
Reg_Fold5_avgMSE = avg(Reg_Fold5_MSE)

'''Regression MSE AE'''
Reg_Fold1_AEMSE = make_reg_AE(Fold1_MSE)
Reg_Fold1_AEavgMSE = avg(Reg_Fold1_AEMSE)

Reg_Fold2_AEMSE = make_reg_AE(Fold2_MSE)
Reg_Fold2_AEavgMSE = avg(Reg_Fold2_AEMSE)

Reg_Fold3_AEMSE = make_reg_AE(Fold3_MSE)
Reg_Fold3_AEavgMSE = avg(Reg_Fold3_AEMSE)

Reg_Fold4_AEMSE = make_reg_AE(Fold4_MSE)
Reg_Fold4_AEavgMSE = avg(Reg_Fold4_AEMSE)

Reg_Fold5_AEMSE = make_reg_AE(Fold5_MSE)
Reg_Fold5_AEavgMSE = avg(Reg_Fold5_AEMSE)

'''Regression MSE No AE'''
Reg_Fold1_noAEMSE = make_reg_noAE(Fold1_MSE)
Reg_Fold1_noAEavgMSE = avg(Reg_Fold1_noAEMSE)

Reg_Fold2_noAEMSE = make_reg_noAE(Fold2_MSE)
Reg_Fold2_noAEavgMSE = avg(Reg_Fold2_noAEMSE)

Reg_Fold3_noAEMSE = make_reg_noAE(Fold3_MSE)
Reg_Fold3_noAEavgMSE = avg(Reg_Fold3_noAEMSE)

Reg_Fold4_noAEMSE = make_reg_noAE(Fold4_MSE)
Reg_Fold4_noAEavgMSE = avg(Reg_Fold4_noAEMSE)

Reg_Fold5_noAEMSE = make_reg_noAE(Fold5_MSE)
Reg_Fold5_noAEavgMSE = avg(Reg_Fold5_noAEMSE)