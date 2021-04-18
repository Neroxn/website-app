import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

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
