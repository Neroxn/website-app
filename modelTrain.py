import numpy as np
import pandas as pd
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet
from sklearn.svm import SVR,SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from werkzeug.utils import redirect
from preprocess import *
from utils import object_encode, onehot_encode, save_user_model, standard_scale
from flask import flash, url_for, session



def create_SVR(selected_parameters):
    """
    Create a SVR model -- y features have to be float/integer
    
    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    kernel = 'rbf' if selected_parameters.get('kernel') == "" else selected_parameters.get('kernel')
    degree = 3 if selected_parameters.get('kernel') == "" else int(selected_parameters.get('degree'))
    C = 1.0 if selected_parameters.get('C') == "" else float(selected_parameters.get('C'))    

    model = SVR(kernel = kernel, degree = degree, C = C)
    return MultiOutputRegressor(model)

def create_SVC(selected_parameters):
    """
    Create a SVC model -- y features have to be integer
    
    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    kernel = 'rbf' if selected_parameters.get('kernel') == "" else selected_parameters.get('kernel')
    degree = 3 if selected_parameters.get('kernel') == "" else int(selected_parameters.get('degree'))
    C = 1.0 if selected_parameters.get('C') == "" else float(selected_parameters.get('C'))

    model = SVC(kernel = kernel, degree = degree, C = C)
    return MultiOutputClassifier(model)

def create_LogisticRegression(selected_parameters):
    """
    Create a SVR model -- y features have to be float/integer
    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    penalty = 'l2' if selected_parameters.get('penalty') == "" else selected_parameters.get('penalty')
    C = 1.0 if selected_parameters.get('C') == "" else float(selected_parameters.get('C'))
    l1_ratio = 0.5 if selected_parameters.get('l1_ratio') == "" else float(selected_parameters.get('l1_ratio'))

    model = LogisticRegression(penalty = penalty, C = C, l1_ratio = l1_ratio)
    return MultiOutputClassifier(model)

def create_LinearRegression(selected_parameters):
    """
    Create a SVR model -- y features have to be float/integer
    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    model = LinearRegression()

    return MultiOutputRegressor(model)

def create_ElasticNet(selected_parameters):
    """
    Create a ElasticNet model which has a mixed model

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    alpha = 1.0 if selected_parameters.get('alpha') == "" else float(selected_parameters.get('alpha'))
    l1_ratio = 0.5 if selected_parameters.get('l1_ratio') == "" else float(selected_parameters.get('l1_ratio'))
    model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
    return MultiOutputRegressor(model)

def create_RandomForestRegressor(selected_parameters):
    """
    Create a RandomForest model for regression

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    n_estimators = 100 if selected_parameters.get('n_estimators') == "" else int(selected_parameters.get('n_estimators'))
    max_depth = None if selected_parameters.get('max_depth') == "" else int(selected_parameters.get('max_depth'))
    min_samples_split = 2 if selected_parameters.get('min_samples_split') == "" else int(selected_parameters.get('min_samples_split'))
    min_samples_leaf = 1 if selected_parameters.get('min_samples_leaf') == "" else int(selected_parameters.get('min_samples_leaf'))


    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)

    return model
    
def create_RandomForestClassifier(selected_parameters):
    """
    Create a RandomForest model --

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    n_estimators = 100 if selected_parameters.get('n_estimators') == "" else int(selected_parameters.get('n_estimators'))
    max_depth = None if selected_parameters.get('max_depth') == "" else int(selected_parameters.get('max_depth'))
    min_samples_split = 2 if selected_parameters.get('min_samples_split') == "" else int(selected_parameters.get('min_samples_split'))
    min_samples_leaf = 1 if selected_parameters.get('min_samples_leaf') == "" else int(selected_parameters.get('min_samples_leaf'))

    model= RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)

    return model

def create_AdaBoostRegressor(selected_parameters):
    """
    Create a AdaBoost model for regression

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    n_estimators =50 if selected_parameters.get('n_estimators') == "" else int(selected_parameters.get('n_estimators'))
    learning_rate = 1.0 if selected_parameters.get('n_estimators') == "" else float(selected_parameters.get('learning_rate'))
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    
    return MultiOutputRegressor(model)

def create_AdaBoostClassifier(selected_parameters):
    """
    Create a AdaBoost model for classification

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    n_estimators =50 if selected_parameters.get('n_estimators') == "" else int(selected_parameters.get('n_estimators'))
    learning_rate = 1.0 if selected_parameters.get('n_estimators') == "" else float(selected_parameters.get('learning_rate'))
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    
    return MultiOutputClassifier(model)
    

def create_CNN(selected_parameters):
    """
    Create a CNN (Convulational Neural Network) model --

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    return None

def create_RNN(selected_parameters):
    """
    Create a RNN (Recurrent Neural Network) model --

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    return None

def create_DNN(seleced_parameters):
    """
    Create a DNN (Deep Neural Network) model --

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    return None 


def model_type(selected_model):
    """
    Return the model type and determine whether the model is a regression/classification/both task.
    << Parameters
    :selected_model: -- Selected model that will be trained

    >> Returns
    :modelType: -- "regression", "classification" or "both", else return flash
    """
    modelType = None
    regression_tasks = ["SVR","LinearRegression","RandomForestRegressor","AdaBoostRegressor"]
    classification_tasks = ["SVC","LogisticRegression","RandomForestClassifier","AdaBoostClassifier"]
    mixed_tasks = []

    if selected_model in regression_tasks:
        modelType = "regression"

    elif selected_model in classification_tasks:
        modelType = "classification"

    elif selected_model in mixed_tasks:
        modelType = "both"

    else:
        flash("An error occured. Please contact the admins!")
    return modelType

def preprocess_for_model(selected_model,X,y):
    """
    Preprocess the input and output to create a working model given as in the form of string:

    << Parameters
    :selected_model: -- the model that is selected by the user via POST method. For every different model
                        different preprocessing may need.
    :X: -- the given X that will be used in the model
    :y: -- the given y that will be used in the model objective

    >> Returns
    :X_processed: -- the after image of the x 
    :y_processed: -- the after image of the y
    """
    concatted_df = pd.concat([X,y],axis = 1)
    numeric_columns = concatted_df.select_dtypes(include=['number']).columns
    object_columns = concatted_df.select_dtypes(include=['object']).columns
    scaler,encoder = None,None
    selected_type = model_type(selected_model)
    # scale Numerical Columns -> Encode Object Columns
    if selected_type == "regression":
        scaled_data,scaler = standard_scale(concatted_df[numeric_columns])
        encoded_data,encoder = object_encode(concatted_df[object_columns])
        if scaler is not None:
            concatted_df[numeric_columns] = scaled_data
        if encoder is not None:
            concatted_df[object_columns] = encoded_data

        X_processed,y_processed =  concatted_df[X.columns],concatted_df[y.columns]

    else:
        scaled_data,scaler = standard_scale(X.select_dtypes(include = ['number']))
        encoded_data,encoder = object_encode(concatted_df[object_columns])
        if scaler is not None:
            concatted_df[X.select_dtypes(include = ['number']).columns] = scaled_data
        if encoder is not None:
            concatted_df[object_columns] = encoded_data

        X_processed,y_processed =  concatted_df[X.columns],concatted_df[y.columns]

    if model_type(selected_model) == "regression": # if model is regression, change integers in y to float
        y_processed_integers = y_processed.select_dtypes(include = [np.int8,np.int16,np.int32,np.int64])
        y_processed[y_processed_integers.columns] = y_processed_integers.astype(np.float32)
        
    if scaler is not None:
        save_user_model(session.get('user_id'),scaler,body = "-model-scaler")
    if encoder is not None:
        save_user_model(session.get('user_id'),encoder,body = "-model-encoder")

    print(y_processed)
    return X_processed,y_processed

def fetch_model(selected_model,selected_parameters):
    """
    Fetch the model by creating a model with selected_model parameter. 
    << Parameters
    :selected_model: -- the model that is selected by the user via POST method. For every different model
                        different preprocessing may need.
    :selected_parameters: -- a dict of selected_parameters that is required in creating model

    >> Returns
    :model: -- model that is created via selected_model and selected_parameters
    """
    if selected_model == "SVR":
        return create_SVR(selected_parameters)

    elif selected_model == "SVC":
        return create_SVC(selected_parameters)

    elif selected_model == "LinearRegression":
        return create_LinearRegression(selected_parameters)

    elif selected_model == "LogisticRegression":
        return create_LogisticRegression(selected_parameters)

    elif selected_model == "RandomForestRegressor":
        return create_RandomForestRegressor(selected_parameters)

    elif selected_model == "RandomForestClassifier":
        return create_RandomForestClassifier(selected_parameters)

    elif selected_model == "AdaBoostRegressor":
        return create_AdaBoostRegressor(selected_parameters)

    elif selected_model == "AdaBoostClassifier":
        return create_AdaBoostClassifier(selected_parameters)


def train_model(model,train_X, train_y):
    """
    Train the model with train_X and train_y
    << Parameters
    :model: -- model that is created and ready for getting input
    :train_X: -- the X that is preprocessed and ready for training the model
    :train_Y: -- the y that is preprocessed and ready for training the model

    >> Returns
    :model: -- model that is trained
    """
    #try:
    #    model.fit(train_X,train_y)
    #except:
    #    flash("An error has occured while training!")
    #    return redirect(url_for("selectAlgo"))
    print(train_X.head(),train_y.head())
    model.fit(train_X,train_y)
    return model

def test_model(model, test_X):
    """
    Test the model with test_X and test_y
    << Parameters
    :model: -- model that is created and ready for getting input
    :test_X: -- the X that is preprocessed and ready for testing the model
    :test_Y: -- the y that is preprocessed and ready for testing the model

    >> Returns
    :model: -- model that is trained
    """
    print(model,model.predict(test_X))
    try:
        predicted_X = model.predict(test_X)
    except:
        flash("An error has occured while testing!")
        return redirect(url_for("selectAlgo"))
    return predicted_X