import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet
from sklearn.svm import SVR,SVC
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from werkzeug.utils import redirect
from .transformers import object_encode, standard_scale
from utils import save_user_model,check_float
from flask import flash, url_for, session



def create_SVR(selected_parameters):
    """
    Create a SVR model -- y features have to be float/integer
    
    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    kernel = 'rbf' if selected_parameters.get('kernel') == False else selected_parameters.get('kernel')
    degree = 3 if check_float(selected_parameters.get('kernel')) == False else int(selected_parameters.get('degree'))
    C = 1.0 if check_float(selected_parameters.get('C')) == False else float(selected_parameters.get('C'))    

    if degree < 0:
        degree = 3
        flash("Negative degree is changed to its default value.")
    if C < 0:
        C = 1.0
        flash("Negative C value is changed to its default value")

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
    degree = 3 if check_float(selected_parameters.get('kernel')) == False else int(selected_parameters.get('degree'))
    C = 1.0 if check_float(selected_parameters.get('C')) == False else float(selected_parameters.get('C'))

    if degree < 0:
        degree = 3
        flash("Negative degree is changed to its default value.")
    if C < 0:
        C = 1.0
        flash("Negative C value is changed to its default value")

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
    C = 1.0 if check_float(selected_parameters.get('C'))  == False else float(selected_parameters.get('C'))
    l1_ratio = 0.5 if check_float(selected_parameters.get('l1_ratio')) == False else float(selected_parameters.get('l1_ratio'))

    if l1_ratio < 0 or l1_ratio > 1:
        l1_ratio = 0.5
        flash("L1 ratio out of boundary and it is changed to its default value.")
    if C < 0:
        C = 1.0
        flash("Negative C value is changed to its default value")

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
    alpha = 1.0 if check_float(selected_parameters.get('alpha'))  == False else float(selected_parameters.get('alpha'))
    l1_ratio = 0.5 if check_float(selected_parameters.get('l1_ratio'))  == False else float(selected_parameters.get('l1_ratio'))
    
    if l1_ratio < 0 or l1_ratio > 1:
        l1_ratio = 0.5
        flash("L1 ratio out of boundary and it is changed to its default value.")
    if alpha < 0:
        alpha = 1.0
        flash("Negative alpha value is changed to its default value")

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
    n_estimators = 100 if check_float(selected_parameters.get('n_estimators')) == False else int(selected_parameters.get('n_estimators'))
    max_depth = None if check_float(selected_parameters.get('max_depth')) == False else int(selected_parameters.get('max_depth'))
    min_samples_split = 2 if check_float(selected_parameters.get('min_samples_split')) == False else int(selected_parameters.get('min_samples_split'))
    min_samples_leaf = 1 if check_float(selected_parameters.get('min_samples_leaf')) == False else int(selected_parameters.get('min_samples_leaf'))
    if n_estimators < 0:
        n_estimators = 100
        flash("Negative n_estimator value is changed to its default value")
    if max_depth < 0:
        max_depth = None
        flash("Negative max_depth value is changed to its default value")
    if min_samples_split < 0:
        min_samples_split = 2
        flash("Negative min_samples_split value is changed to its default value")
    if min_samples_leaf < 0:
        min_samples_leaf = 1
        flash("Negative min_samples_leaf value is changed to its default value")

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
    n_estimators = 100 if check_float(selected_parameters.get('n_estimators')) == False else int(selected_parameters.get('n_estimators'))
    max_depth = None if check_float(selected_parameters.get('max_depth')) == False else int(selected_parameters.get('max_depth'))
    min_samples_split = 2 if check_float(selected_parameters.get('min_samples_split')) == False else int(selected_parameters.get('min_samples_split'))
    min_samples_leaf = 1 if check_float(selected_parameters.get('min_samples_leaf')) == False else int(selected_parameters.get('min_samples_leaf'))

    if n_estimators < 0:
        n_estimators = 100
        flash("Negative n_estimator value is changed to its default value")
    if max_depth < 0:
        max_depth = None
        flash("Negative max_depth value is changed to its default value")
    if min_samples_split < 0:
        min_samples_split = 2
        flash("Negative min_samples_split value is changed to its default value")
    if min_samples_leaf < 0:
        min_samples_leaf = 1
        flash("Negative min_samples_leaf value is changed to its default value")

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
    n_estimators =50 if check_float(selected_parameters.get('n_estimators')) == False else int(selected_parameters.get('n_estimators'))
    learning_rate = 1.0 if check_float(selected_parameters.get('n_estimators')) == False else float(selected_parameters.get('learning_rate'))
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    
    if n_estimators < 0:
        n_estimators = 50
        flash("Negative n_estimator value is changed to its default value")
    if learning_rate < 0:
        learning_rate = 1.0
        flash("Negative learning_rate value is changed to its default value")
        
    return MultiOutputRegressor(model)

def create_AdaBoostClassifier(selected_parameters):
    """
    Create a AdaBoost model for classification

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    n_estimators =50 if check_float(selected_parameters.get('n_estimators')) == False else int(selected_parameters.get('n_estimators'))
    learning_rate = 1.0 if check_float(selected_parameters.get('n_estimators')) == False else float(selected_parameters.get('learning_rate'))
    
    if n_estimators < 0:
        n_estimators = 50
        flash("Negative n_estimator value is changed to its default value")
    if learning_rate < 0:
        learning_rate = 1.0
        flash("Negative learning_rate value is changed to its default value")

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
    scaler,encoder = None,None
    selected_type = model_type(selected_model)
    scalers,encoders = [],[]

    scaled_data_X,scaler_X = standard_scale(X.select_dtypes(include = "number"))
    session["numerical_X"] = [col for col in X.select_dtypes(include = "number").columns]

    if selected_type == "classification":
        scaled_data_y,scaler_y = y.select_dtypes(include = [np.int8,np.int16,np.int32,np.int64]),None
        session["numerical_y"] = [col for col in y.select_dtypes(include =[np.int8,np.int16,np.int32,np.int64]).columns]
    else:
        scaled_data_y,scaler_y = standard_scale(y.select_dtypes(include = "number"))
        session["numerical_y"] = [col for col in y.select_dtypes(include = "number").columns]
    



    encoded_data_X,encoder_X = object_encode(X.select_dtypes(include = "object"))
    session["object_X"] = [col for col in X.select_dtypes(include = "object").columns]

    encoded_data_y,encoder_y = object_encode(y.select_dtypes(include = "object"))
    session["object_y"] = [col for col in y.select_dtypes(include = "object").columns]

    scalers += [scaler_X]
    if scaler_X is not None:
        X[X.select_dtypes(include = "number").columns] = scaled_data_X

    scalers += [scaler_y]
    if scaler_y is not None:
        y[y.select_dtypes(include = "number").columns] = scaled_data_y

    encoders += [encoder_X]
    if encoder_X is not None:
        X[X.select_dtypes(include = "object").columns] = encoded_data_X    

    encoders += [encoder_y]
    if encoder_y is not None:
        y[y.select_dtypes(include = "object").columns] = encoded_data_y   

    X_processed,y_processed =  X,y

    if selected_type == "regression": # if model is regression, change integers in y to float
        y_processed_integers = y_processed.select_dtypes(include = [np.int8,np.int16,np.int32,np.int64])
        y_processed[y_processed_integers.columns] = y_processed_integers.astype(np.float32)
        
    save_user_model(encoders,session.get('user_id'),body = "-model-encoders")
    save_user_model(scalers,session.get('user_id'),body = "-model-scalers")

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
    try:
        predicted_X = model.predict(test_X)
    except:
        flash("An error has occured while testing!")
        return redirect(url_for("selectAlgo"))
    return predicted_X