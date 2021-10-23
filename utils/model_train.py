from flask.templating import render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet
from sklearn.svm import SVR,SVC
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from werkzeug.utils import redirect
from .transformers import min_max_scale, object_encode, onehot_encode, standard_scale
from utils import save_user_model,check_float
from utils.transformers import calculate_model_score,pixel_scaler
from flask import flash, url_for, session
import tensorflow as tf



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
    if penalty == "elasticnet":
        solver = 'saga'
    else:
        solver = "liblinear"
    model = LogisticRegression(penalty = penalty, C = C, l1_ratio = l1_ratio, solver=solver)
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
    if max_depth and max_depth < 0:
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
    if max_depth and max_depth < 0:
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
    learning_rate = 1.0 if check_float(selected_parameters.get('learning_rate')) == False else float(selected_parameters.get('learning_rate'))
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
    learning_rate = 1.0 if check_float(selected_parameters.get('learning_rate')) == False else float(selected_parameters.get('learning_rate'))
    
    if n_estimators < 0:
        n_estimators = 50
        flash("Negative n_estimator value is changed to its default value")
    if learning_rate < 0:
        learning_rate = 1.0
        flash("Negative learning_rate value is changed to its default value")

    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    
    return MultiOutputClassifier(model)
    

def get_layer(information_dict, input_shape = None):
    """
    Construct a layer and return that layer for creating a model.
    :information_dict: -- dictionary that holds information for layers
    """
    layer_name = information_dict["layer_name"]
    if layer_name == "Dense": # create dense layer
        layer_size = information_dict["units"]
        layer_activation = information_dict["activation"]
        layer = tf.keras.layers.Dense(layer_size,layer_activation)

    elif layer_name == "Dropout": # create dropout layer
        rate = information_dict["ratio"]
        layer = tf.keras.layers.Dropout(rate)

    elif layer_name == "BatchNorm": # create batch normalization layer
        momentum = information_dict["momentum"]
        epsilon = information_dict["epsilon"]
        layer = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        
    elif layer_name == "LSTM":
        units = information_dict["units"]
        activation = information_dict["activation"]
        layer = tf.keras.layers.LSTM(units = units, activation = activation)

    elif layer_name == "Conv2D":
        filters = information_dict["units"]
        kernel_size = information_dict["size"]
        strides = information_dict["strides"]
        activation = information_dict["activation"]
        layer = tf.keras.layers.Conv2D(filters = filters, kernel_size = (kernel_size,kernel_size),
        strides = (strides,strides), activation= activation)

    elif layer_name == "MaxPooling2D":
        pool_size = information_dict["size"]
        strides = information_dict["strides"]
        layer = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

    elif layer_name == "Flatten":
        layer = tf.keras.layers.Flatten()

    else:
        flash("An error occured while constructing layers!")
        return None # redirect
    print("Layer returned : ",layer)
    return layer
def loss_fetcher(**kwargs):
    if kwargs["loss"] == "sparse_categorical_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    elif kwargs["loss"] == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    elif kwargs["loss"] == "mse":
        return tf.keras.losses.MeanSquaredError()

    elif kwargs["loss"] == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    
    else: # this will not be achiaveble 
        flash("An error has occured while selecting loss function!")
        return render_template(url_for("selectAlgo_DL"))

def create_RNN(selected_parameters):
    """
    Create a RNN (Recurrent Neural Network) model --

    << Parameters
    :selected_parameters: -- parameters that is selected to construct the model

    >> Returns
    :model: -- model that is created
    """
    # TODO : Using @selected_parameters, initialize a set of layers.
    # TODO : Create a function that takes these parameters and outputs a model by constructing them.
    return None

def create_DNN(seleced_parameters, model_configurations, final_activation = "linear", **kwargs):
    """
    Create a DNN (Deep Neural Network) model --

    << Parameters
    :selected_parameters: -- layers that is selected to construct the model
        >@selected_parameters is in the format of list of dictionary 
        (number of )
    :model_configurations: -- general parameters (loss, optimizer, etc) for the model
    :final_activation: -- the final activation function for the final layer
    >> Returns
    :model: -- model that is created
    """
    add_flatten = True
    input_size = len(session.get('selected_x'))
    type_of_model = model_type(session.get("selected_model"))
    if type_of_model == "regression":
        output_size = len(session.get("selected_y")) # --> else no of unique values.
    else:
        if model_configurations["loss"] == "binary_crossentropy":
            output_size = 1
        elif model_configurations["loss"] == "sparse_categorical_crossentropy":
            output_size = kwargs.get("no_of_unique")

    model = tf.keras.Sequential() # base model 

    if session.get("selected_model") in ["DNN_R","DNN_C"]: # 
        model.add(tf.keras.Input(shape  = input_size))
        add_flatten = False

    for layer_info in seleced_parameters: # user defined layers
        try:
            model.add(get_layer(layer_info))
        except:
            flash("An error made while creating the layer : {}",layer_info)
    
    # output layer, if regression output linear else softmax
    if add_flatten:
        model.add(get_layer({"layer_name" : "Flatten"}))
    model.add(tf.keras.layers.Dense(output_size, activation = final_activation))
    
    # general configurations
    optimizer = model_configurations["optimizer"]
    loss = loss_fetcher(loss = model_configurations["loss"])
    metrics = model_configurations["metrics"]
    model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
    return model


def model_type(selected_model):
    """
    Return the model type and determine whether the model is a regression/classification/both task.
    << Parameters
    :selected_model: -- Selected model that will be trained

    >> Returns
    :modelType: -- "regression", "classification" or "both", else return flash
    """
    # might resolve some errors
    
    modelType = None
    regression_tasks = ["SVR","LinearRegression","RandomForestRegressor","AdaBoostRegressor","DNN_R","CNN_R"]
    classification_tasks = ["SVC","LogisticRegression","RandomForestClassifier","AdaBoostClassifier","DNN_C","CNN_C"]

    if selected_model in regression_tasks:
        modelType = "regression"

    elif selected_model in classification_tasks:
        modelType = "classification"
    else:
        flash("An error occured. Please contact the admins!")
    return modelType

def cross_validate_models(X,y,model,K,**kwargs):
    """
    Use cross validation using the constructed model.

    Return:
    :models: -- trained models
    :model_scores: -- list that holds dictionary for each model with loss scores on test data
    """
    # calculate the sizes for test and train
    if session.get("selected_model") in ["CNN_C","CNN_R","DNN_R","DNN_C"]:
        untrained_model = tf.keras.models.clone_model(model)
    else:
        untrained_model = clone(model)
    full_size = X.shape[0]
    test_size = int(full_size/K) 
    type_of_model = model_type(session.get("selected_model"))

    # initialize emtpy lists
    models = []
    model_scores = []
    test_predictions = ([],[])
    input_shape = (None,None,None)

    X_copy = X.copy()
    y_copy = y.copy()

    if type_of_model == "classification" and len(session.get('selected_y')) == 1: # meaningless for regression model
        fold = StratifiedKFold(n_splits=K,shuffle=True)

    else:
        fold = KFold(n_splits=K,shuffle=True)

    for train_index,test_index in fold.split(X_copy,y_copy):

        X_train, X_test = X_copy.iloc[train_index], X_copy.iloc[test_index]
        y_train, y_test = y_copy.iloc[train_index], y_copy.iloc[test_index]

    
        print("Check dimensions : ",X_test.shape,X_train.shape)
        # train the model
        if session.get("selected_model") in ["CNN_R","CNN_C"]: # if model is CNN, reshape inputs
            input_shape = kwargs["input_shape"]
            X_train = X_train.values.reshape(X_train.shape[0],input_shape[0],input_shape[1],input_shape[2])
            X_test = X_test.values.reshape(X_test.shape[0],input_shape[0],input_shape[1],input_shape[2])

        if session.get("selected_model") in ["CNN_R","CNN_C","DNN_R","DNN_C"]:
            model = train_DNN(model,X_train,y_train, epochs= kwargs["epochs"], batch_size = kwargs["batch_size"],
            callbacks = kwargs["callbacks"])
        else:   
            model = train_model(model,X_train,y_train)

        if model is None: # if an error occured while training the model 
            break

        model_score,test_predictions = calculate_model_score(model,X_test.copy(),y_test.copy(), type_of_model = type_of_model,input_shape = input_shape) # calculate model scores that are useful

        # append result to list
        models += [model]
        model_scores += [model_score]

        # reset back to default model
        if session.get("selected_model") in ["CNN_C","CNN_R","DNN_R","DNN_C"]:
            model = tf.keras.models.clone_model(untrained_model)
            model.compile(loss = loss_fetcher(loss = kwargs["loss"]), metrics = kwargs["metrics"], optimizer = kwargs["optimizer"])
        else:
            model = clone(untrained_model)

    return model_scores,models,test_predictions

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
    print(selected_model)
    # if selected model is CNN, divide the pixels by 255.0 instead of scaling
    if selected_model in ["CNN_R","CNN_C"]:
        scaled_data_X,scaler_X = min_max_scale(X.select_dtypes(include = "number"))
        session["numerical_X"] = [col for col in X.select_dtypes(include = "number").columns]
    else:
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
    
    print(X_processed,y_processed)
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

    elif selected_model == "RNN":
        return create_RNN(selected_parameters)

    elif selected_model == "DNN":
        return create_DNN(selected_parameters)

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
    try:
        model.fit(train_X,train_y)
    except:
        flash("Something went wrong! Constructed model is not valid for the selected data.")
        return None
    return model

def train_DNN(model,train_X,train_y, epochs = 30, batch_size = 32,callbacks = []):
    """
    Train the model with train_X and train_y
    << Parameters
    :model: -- model that is created and ready for getting input
    :train_X: -- the X that is preprocessed and ready for training the model
    :train_Y: -- the y that is preprocessed and ready for training the model

    >> Returns
    :model: -- model that is trained
    """
    try:
        model.fit(train_X,train_y, epochs = epochs,
        callbacks = callbacks, batch_size = batch_size)
    except:
        flash("Something went wrong! Constructed model is not valid for the selected data.")
        return None
    return model

################################################################################################################################
# TODO : Function for fitting neural networks                                                                                  #
# TODO : If RNN, assume that there is a ID column so that we can use group_by. Also assume that Date/Sequential data is sorted.#                                                                                    #
################################################################################################################################
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