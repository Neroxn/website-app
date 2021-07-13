
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from flask import Flask, request, redirect, url_for,render_template,session, flash
from utils import load_user_model,save_user_model,concat_columns

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
        df[col] = encoder.fit_transform(df[col])
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
    print("Encoded columns are : ",encoded_columns)
    print(df)
    if object_prefix is not None:
        encoded_columns = [object_prefix + col for col in encoded_columns]
    return pd.DataFrame(df,columns = encoded_columns),encoder


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
            print(test_y[test_col].shape)
            print(predicted_y[pred_col].shape)
            print(len(np.unique(test_y[test_col])),len(np.unique(predicted_y[pred_col])))
            print(np.unique(test_y[test_col]),np.unique(predicted_y[pred_col]))
            f1_scores += [f1_score(test_y[test_col],predicted_y[pred_col],average="macro")] 

    return model_scores,mse_errors,mae_errors,log_errors,f1_scores
 
def apply_model_transformers(X,y):
    """
    Apply encoders and scalers that are saved to dataframe.
    << Parameters
    :X: -- X dataframe that we will revert back to the original values by using scaler
    :y: -- y dataframe that we will revert back to the  original values by using scaler
    """
    X = X.copy()
    y = y.copy()
    encoders = load_user_model(session.get('user_id'),body = "-model-encoders")
    scalers = load_user_model(session.get('user_id'),body = "-model-scalers")
    print(encoders,scalers)
    scaler_X,encoder_X = scalers[0],encoders[0]
    scaler_y,encoder_y= scalers[1],encoders[1]

    if scaler_X is not None:
        print(session.get("numerical_X"))
        X[session.get("numerical_X")] = scaler_X.transform(X[session.get("numerical_X")])
    if scaler_y is not None:
        print(session.get("numerical_y"))
        y[session.get("numerical_y")] = scaler_y.transform(y[session.get("numerical_y")])
    if encoder_X is not None:
        for index,col in enumerate(session.get("object_X")):
            print(col,index,encoder_X[index])
            X[col] = encoder_X[index].transform(X[col])
    if encoder_y is not None:
        for index,col in enumerate(session.get("object_y")):
            print(col,index,encoder_y[index])
            y[col] = encoder_y[index].transform(y[col])
            
    return X,y
        


def revert_model_transformers(X,y):
    """
    Apply encoders and scalers that are saved to dataframe.
    << Parameters
    :X: -- X dataframe that we will revert back to the original values by using scaler
    :y: -- y dataframe that we will revert back to the  original values by using scaler

    >> Returns
    :X: -- X value that is scaled 
    :y: -- y value that is scaled
    """
    X = X.copy()
    y = y.copy()
    encoders = load_user_model(session.get('user_id'),body = "-model-encoders")
    scalers = load_user_model(session.get('user_id'),body = "-model-scalers")
    print(encoders,scalers)
    scaler_X,encoder_X = scalers[0],encoders[0]
    scaler_y,encoder_y= scalers[1],encoders[1]

    if scaler_X is not None:
        print(session.get("numerical_X"))
        X[session.get("numerical_X")] = scaler_X.inverse_transform(X[session.get("numerical_X")])
    if scaler_y is not None:
        print(session.get("numerical_y"))
        y[session.get("numerical_y")] = scaler_y.inverse_transform(y[session.get("numerical_y")])

    if encoder_X is not None:
        for index,col in enumerate(session.get("object_X")):
            print(col,index,encoder_X[index])
            print(X[col].head())
            X[col] = encoder_X[index].inverse_transform(X[col])
    if encoder_y is not None:
        for index,col in enumerate(session.get("object_y")):
            print(col,index,encoder_y[index])
            print(y[col].head())
            y[col] = encoder_y[index].inverse_transform(y[col])
            
    return X,y

def apply_user_transformers(df,model_type):
    """
    Apply transformations that are done by user
    """
    return None

def revert_user_transformers():
    """
    Revert user transformers
    """
    return None

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
    selected_df = data[selected_columns]

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

    elif mode == "onehot-encode":
        #Check if columns are discreate or object
        selected_df = selected_df.select_dtypes(exclude = ["float32","float64"])
        selected_columns = selected_df.columns
        encoded_df, encoder = onehot_encode(selected_df)
        for col in encoded_df.columns:
            data[col] = encoded_df[col]
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
        imputed_df = imputer.fit_transform(selected_df)
        data[selected_columns] = imputed_df


    elif mode == "impute-columns-mean":
        imputer = SimpleImputer(strategy = "median")
        selected_df = selected_df.select_dtypes(include = ["number"])
        selected_columns = selected_df.columns
        imputed_df = imputer.fit_transform(selected_df)
        data[selected_columns] = imputed_df

    
    elif mode == "impute-columns-mfq":
        imputer = SimpleImputer(strategy = "most_frequent")
        imputed_df = imputer.fit_transform(selected_df)
        data[selected_columns] = imputed_df


    if delete_column and mode in ["sum","mean","difference","concat"]:
        return data.drop(selected_columns,axis = 1),None
    
    return data,None
