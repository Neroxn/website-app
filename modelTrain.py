import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from preprocess import *
def applySVM(xTrain, yTrain, kernel= 'linear', c= 1, gamma= 'auto', degree= 4):
    """
    Applies SVM on training set and return model.
    
    if hyperparameters are not given, default values of SVM are used.
    
    Detail about hyperparameters:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    """
    
    if(kernel == 'linear'):
        model = SVR(kernel= kernel, C=c, cache_size= 1000)
    elif(kernel=='poly'):
        model = SVR(kernel= kernel, C=c, gamma= gamma, cache_size= 1000, degree= degree)
    else:
        model = SVR(kernel= kernel, C=c, gamma= gamma, cache_size= 1000)
    
    model.fit(xTrain, yTrain)
    return model

def applyRandomForest(xTrain, yTrain, numberEstimator= 100, maxDepth= None, minSamplesLeaf= 1):
    """
    Applies RandomForestRegressor on training set and return model.
    
    if hyperparameters are not given, default values RFR are used.
    
    Detail about hyperparameters:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    """
    model = RandomForestRegressor(n_estimators= numberEstimator, max_depth= maxDepth, min_samples_leaf= minSamplesLeaf)
    
    model.fit(xTrain, yTrain)
    return model

def applyAdaBoost(xTrain, yTrain, numberEstimator= 50, learningRate= 1, loss= 'linear'):
    """
    Applies AdaBoostRegressor on training set and return model.
    
    if hyperparameters are not given, default values AdaBoostRegressor are used.
    
    Detail about hyperparameters:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
    """
    model = AdaBoostRegressor(n_estimators= numberEstimator, learning_rate= learningRate, loss=loss)
    
    model.fit(xTrain, yTrain)
    return model

"""
def applyLSTM(df, selectedX, selectedY):
    
    #Convert DateTime
    df['DateTime'] = pd.to_datetime(df['Date.Time'])
    
    #Drop NaN and duplicate values
    df = dropNanAndDuplicates(df, 0.75)
    
    #Save orginal df
    df2 = df.drop(['ID', 'Date.Time', 'Eye', 'idEye'], axis = 1)
    
    #Scale
    df2, scaler, scalerY = scale(df2, selectedY)
    
    #Backup ID columns from orginal df
    df2['ID'] = df['ID']
    df2['Date.Time'] = df['Date.Time']
    df2['Eye'] = df['Eye']
    df2['idEye'] = df['idEye']
    
    #Generators
    def train_generator():
        while True:
            x_train, y_train = shuffle_list(x_tt, y_tt)
            for i in range(x_train.shape[0]):
                train_x = x_train[i].reshape((1, -1, len(selectedX)))
                train_y = y_train[i].reshape((1, -1, len(selectedY)))
                yield train_x, train_y
                
    def valid_generator():
        while True:
            x_train, y_train = shuffle_list(x_valid, y_valid)
            for i in range(x_train.shape[0]):
                train_x = x_train[i].reshape((1, -1, len(selectedX)))
                train_y = y_train[i].reshape((1, -1, len(selectedY)))
                yield train_x, train_y
    
    #fetch dataFrame
    (x_org, y_org), (x_left, y_left), (x_right, y_right) = fetch_ID_dataframe_last(df2, min_observation=1, seperate=True)
"""