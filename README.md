__This is a current work in progression so many files are missing__

# Machine and Deep Learning for Stock Price Prediction: Comparison of Classification and Regression Techniques

The objective of this report is to compare the use of classification models and regression models from Machine and Deep Learning are used to predict the price trend of a stock. In this report the stock that is used is the Koninklijke KPN N.V. (KPN) stock from the Amsterdam Exchange (AEX) Index. Technical features and the price of 4 other stocks from the AEX Index are used to train 12 models. The 12 models are 6 regression and 6 classification models, these are Support Vector Machines/Regression, Feedfor- ward Neural Networks, Recurrent Neural Networks and auto-encoded variants of those models. Each model trained to tackle the classification problem and the regression problem. It is found on average that classification models are able to predict price movements better, however this comes at the cost of only achieving higher results when the models are trained on more data, regression models actually outperformed classification models when less training data is used. However, it is also found that re- gression models can accurately predict the actual price with an average mean absolute error as low as 0.0867 which could be of more value than trend prediction to certain investors. Specific models are then compared Overall it is concluded that classification models work best in predicting trend with the best trend prediction model being a Recurrent Neural Network which predicted the correct trend 55.46%.

---

## Table of Contents:
* [Report](addlink)
* Classification Models:
    - [SVM](SVM.py)
    - [MLP](MLP%20-%20Classification.py)
    - [RNN](RNN%20-%20Classification.py)
    - [Auto-Encoded SVM](add soon)
    - [Auto-Encoded MLP](add soon)
    - [Auto-Encoded RNN](add soon)
* Regression Models:
    - [SVM](SVR.py)
    - [MLP](MLP%20-%20Regression.py)
    - [RNN](RNN%20-%20Regression.py)
    - [Auto-Encoded SVM](add soon)
    - [Auto-Encoded MLP](add soon)
    - [Auto-Encoded RNN](add soon)
