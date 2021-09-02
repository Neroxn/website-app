
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from flask import Flask, request, redirect, url_for,render_template,session, flash
from utils import load_user_model,save_user_model,concat_columns
from .graphs import confusion_matrix_plot

def min_max_scale(df,object_prefix = None):
    """
    :df: -- is the dataframe with values that will be scaled
    :object_prefix: -- if provided, add prefix to every column name
    """
    if df.empty:
        return df,None
    scaler = MinMaxScaler()
    original_columns = df.columns
    df = scaler.fit_transform(df)

    if object_prefix is not None:
        original_columns = [object_prefix + col for col in original_columns]

    return pd.DataFrame(df,columns = original_columns),scaler

def standard_scale(df,object_prefix = None):
    """
    Inputs:
    :df: -- is the dataframe with values that will be scaled
    :object_prefix: -- if provided, add prefix to every column name

    Returns:
    :dataframe object: -- Dataframe object with scaled columns
    :scaler: -- Scaler used in scaling.
    """
    if df.empty:
        return df,None
    scaler = StandardScaler()
    original_columns = df.columns
    try:
        df = scaler.fit_transform(df)
    except: 
        flash("An error has occured!")
        return df,None
        
    if object_prefix is not None:
        original_columns = [object_prefix + col for col in original_columns]

    return pd.DataFrame(df,columns = original_columns),scaler

def object_encode(df,object_prefix = None):
    """
    :df: -- is the dataframe with values that will be scaled
    :object_prefix: -- if provided, add prefix to every column name
    """
    encoders= []
    if df.empty:
        return df,None

    original_columns = df.columns
    for col in df.columns:
        encoder = LabelEncoder()
        print("Encoding the column : ",col)
        df[col] = encoder.fit_transform(df[col].astype(str))
        print(df[col])
        encoders += [encoder]

    
    if object_prefix is not None:
        original_columns = [object_prefix + col for col in original_columns]
    return pd.DataFrame(df,columns = original_columns),encoders

def onehot_encode(df,object_prefix = None):
    """
    :df: -- is the dataframe with values that will be scaled
    :object_prefix: -- if provided, add prefix to every column name
    """
    encoder = OneHotEncoder()
    original_columns = df.columns
    df = encoder.fit_transform(df).toarray()
    encoded_columns = encoder.get_feature_names(original_columns)
    if object_prefix is not None:
        encoded_columns = [object_prefix + col for col in encoded_columns]
    return pd.DataFrame(df,columns = encoded_columns),encoder

def pixel_scaler(X):
    X_copy = X.copy()
    scaled_pixels = (X_copy)/255.0
    return scaled_pixels

def get_metrics(typeModel,test_y,predicted_y):
    """
    Calculate metrics that will be displayed in the result screen
    :typeModel: -- Is the type of the model that we have selected as the task
    :test_y: -- Y values of the test data.
    :predicted_y: -- Y values that are predicted from the test data.
    """
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,f1_score,log_loss,confusion_matrix
    model_scores,mse_errors,mae_errors = [],[],[]
    log_errors,f1_scores = [],[]
    if typeModel == "regression":
        for test_col,pred_col in zip(test_y.columns,predicted_y.columns):
            model_scores += [r2_score(test_y[test_col],predicted_y[pred_col])]
            mse_errors += [mean_squared_error(test_y[test_col],predicted_y[pred_col])]
            mae_errors += [mean_absolute_error(test_y[test_col],predicted_y[pred_col])]

    elif typeModel == "classification":
        for test_col,pred_col in zip(test_y.columns,predicted_y.columns):   
            model_scores += [np.mean(test_y[test_col].values == predicted_y[pred_col].values)]
            f1_scores += [f1_score(test_y[test_col],predicted_y[pred_col],average="macro")] 

    return model_scores,mse_errors,mae_errors,log_errors,f1_scores
 
def apply_model_transformers(X = None,y = None):
    """
    Apply encoders and scalers that are saved to dataframe.
    << Parameters
    :X: -- X dataframe that we will revert back to the original values by using scaler
    :y: -- y dataframe that we will revert back to the  original values by using scaler
    """
    if X is not None:
        X = X.copy()
    if y is not None:
        y = y.copy()
        
    encoders = load_user_model(session.get('user_id'),body = "-model-encoders")
    scalers = load_user_model(session.get('user_id'),body = "-model-scalers")
    scaler_X,encoder_X = scalers[0],encoders[0]
    scaler_y,encoder_y= scalers[1],encoders[1]

    if scaler_X is not None and X is not None:
        X[session.get("numerical_X")] = scaler_X.transform(X[session.get("numerical_X")])
    if scaler_y is not None and y is not None:
        y[session.get("numerical_y")] = scaler_y.transform(y[session.get("numerical_y")])
    if encoder_X is not None and X is not None:
        for index,col in enumerate(session.get("object_X")):
            X[col] = encoder_X[index].transform(X[col])
    if encoder_y is not None and y is not None:
        for index,col in enumerate(session.get("object_y")):
            y[col] = encoder_y[index].transform(y[col])
            
    return X,y
        


def revert_model_transformers(X = None,y = None):
    """
    Apply encoders and scalers that are saved to dataframe.
    << Parameters
    :X: -- X dataframe that we will revert back to the original values by using scaler
    :y: -- y dataframe that we will revert back to the  original values by using scaler

    >> Returns
    :X: -- X value that is scaled 
    :y: -- y value that is scaled
    """
    if X is not None:
        X = X.copy()
    if y is not None:
        y = y.copy()
    encoders = load_user_model(session.get('user_id'),body = "-model-encoders")
    scalers = load_user_model(session.get('user_id'),body = "-model-scalers")
    scaler_X,encoder_X = scalers[0],encoders[0]
    scaler_y,encoder_y= scalers[1],encoders[1]

    if scaler_X is not None and X is not None:
        X[session.get("numerical_X")] = scaler_X.inverse_transform(X[session.get("numerical_X")])
    if scaler_y is not None and y is not None:
        y[session.get("numerical_y")] = scaler_y.inverse_transform(y[session.get("numerical_y")])

    if encoder_X is not None and X is not None:
        for index,col in enumerate(session.get("object_X")):

            X[col] = encoder_X[index].inverse_transform(X[col])
    if encoder_y is not None and y is not None:
        for index,col in enumerate(session.get("object_y")):

            y[col] = encoder_y[index].inverse_transform(y[col])
            
    return X,y

def combine_columns(data, selected_columns, new_column_name, mode, delete_column = True):
    """
    Parameters:
    :data: -- Dataframe that is given
    :selected_columns: -- Subset of the features that will be combined into a single column.
    :new_column_name: -- Name of the new column:
    :mode: -- Which operation will be used
        *) mode = mean -- sum the columns and take avarage of it
        *) mode = sum -- sum the columns
        *) mode = differnce -- take the difference of the columns. Two columns must be selected.
        *) mode = concat -- concat the object columns. An example could be [M,F],[Left,Right] -> MLeft,MRight,FLeft,FRight

    :delete_column: -- If true, discard the used columns. 
    """
    not_entirely_nan = data[selected_columns].isna().sum() != data.shape[0] # columns that has a valid number 

    if not_entirely_nan.all() != True:
        flash("Feature {} has no valid value!".format(list(data[selected_columns].columns[np.invert(not_entirely_nan)])))

    selected_columns = data[selected_columns].columns[not_entirely_nan]  
    selected_df = data[selected_columns]
    selected_df = selected_df.fillna(value = np.nan)

    if mode == "mean":
        data[new_column_name] = selected_df.sum(axis = 1)/len(selected_columns)
        
    elif mode == "sum":
        data[new_column_name] = selected_df.sum(axis = 1)

    elif mode == "difference":
        data[new_column_name] = selected_df.iloc[:,0] - selected_df.iloc[:,1]

    elif mode == "concat": 
        concated_columns = concat_columns(data,selected_columns)
        data[new_column_name] = concated_columns 

    elif mode == "drop-nan-rows":
        data = selected_df.dropna()

    elif mode == "drop-nan-columns":
        data = selected_df.dropna(axis=1)

    elif mode == "min-max-scale":
        #Check if columns are numeric
        selected_df = selected_df.select_dtypes(include = ["number"])
        selected_columns = selected_df.columns
        scaled_df,scaler = min_max_scale(selected_df)
        data[selected_columns] = scaled_df
        return data,scaler

    elif mode == "object-encode":
        #Check if columns are object
        selected_df = selected_df.select_dtypes(include = ["object"])
        selected_columns = selected_df.columns
        encoded_df,encoder = object_encode(selected_df)
        data[selected_columns] = encoded_df
        return data,encoder


    elif mode == "drop-columns":
        data.drop(selected_columns,axis=1,inplace=True), 
        
    elif mode == "standard-scale":
        #Check if columns are numeric
        selected_df = selected_df.select_dtypes(include = ["number"])
        selected_columns = selected_df.columns
        scaled_df, scaler = standard_scale(selected_columns)
        data[selected_columns] = scaled_df
        return data,scaler

    elif mode == "impute-columns-median": 
        
        imputer = SimpleImputer(strategy = "mean")
        selected_df = selected_df.select_dtypes(include = ["number"])
        selected_columns = selected_df.columns
        imputed_df = imputer.fit_transform(selected_df.values)
        data[selected_columns] = imputed_df


    elif mode == "impute-columns-mean":
        imputer = SimpleImputer(strategy = "median")
        selected_df = selected_df.select_dtypes(include = ["number"])
        selected_columns = selected_df.columns
        imputed_df = imputer.fit_transform(selected_df.values)
        data[selected_columns] = imputed_df

    
    elif mode == "impute-columns-mfq":
        imputer = SimpleImputer(strategy = "most_frequent")
        imputed_df = imputer.fit_transform(selected_df.values)
        data[selected_columns] = imputed_df

    if delete_column and mode in ["sum","mean","difference","concat"]:
        flash("Operation successfull!")
        return data.drop(selected_columns,axis = 1),None
    
    flash("Operation successfull!")
    return data,None

def calculate_model_score(model,test_X,test_y, type_of_model = None, **kwargs):

    # get predicted dataframe 
    predicted_y = model.predict(test_X)
    if session.get("selected_model") in ["CNN_C","DNN_C"]: # if task is classification
        predicted_y = np.argmax(predicted_y,axis=1)

    predicted_y = pd.DataFrame(predicted_y,columns = session.get("selected_y"))
    predicted_y.set_index([test_y.index],inplace=True)
    scores = {}

    # if CNN was used, revert back to original form 
    if session.get("selected_model") in ["CNN_C","CNN_R"]:
        test_X_reshaped = test_X.reshape((test_X.shape[0],-1)) 
        test_X = pd.DataFrame(test_X_reshaped, columns = session.get("selected_x"))

    if type_of_model == "regression":
        # revert results before calculating metrics. Transformers are used in preprocessing the data.

        _,test_y = revert_model_transformers(test_X,test_y) 
        _,predicted_y = revert_model_transformers(test_X,predicted_y)
        
        # change column names for readability
        predicted_y.columns = ["predicted_" + col for col in test_y.columns]
        test_y.columns = ["actual_" + col for col in test_y.columns]
        result_dataframe = pd.concat([test_y,predicted_y],axis=1)   
        
        # calculate the metrics that will be displayed
        model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,test_y,predicted_y)

    else:
        # calculate the metrics that will be displayed
        model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,test_y,predicted_y)
        
        # revert results before calculating metrics. Transformers are used in preprocessing the data.
        _,test_y = revert_model_transformers(test_X,test_y)
        _,predicted_y = revert_model_transformers(test_X,predicted_y)
                
        predicted_y.columns = ["predicted_" + col for col in test_y.columns]
        test_y.columns = ["actual_" + col for col in test_y.columns]
        result_dataframe = pd.concat([test_y,predicted_y],axis=1)  

    # currently used metrics
    scores["model_scores"] = model_scores
    scores["mse_errors"] = mse_errors
    scores["mae_errors"] = mae_errors
    scores["log_errors"] = log_errors
    scores["f1_scores"] = f1_scores
    
    return scores,(test_y,predicted_y)
