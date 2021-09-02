from flask import Flask, request, redirect, url_for,render_template,session, flash
import numpy as np
import pandas as pd
import os
import shutil
import pickle
import tensorflow as tf
from numpy import arange
from collections import OrderedDict
from sklearn.impute import SimpleImputer


ALLOWED_EXTENSIONS = set(['txt', 'csv'])

def load_dataset(path,delimitter,qualifier,assumption = False):
    """
    Read file with given extension.
    If CSV : Read columns with pandas
    if TXT : Read columns manually

    Parameters:
    :path: -- path to the txt file
    :delimitter: -- delimitter for splitting the columns/values
    :assumption: -- if true, assume first line holds the information about valueTypes

    Returns:
    :dataTypes: - return dataTypes list
    :dataColumns: -- return dataColumns list
    :df: -- return dataframe of the file
    """
    if delimitter == "":
        delimitter = ","
    if qualifier == "":
        qualifier = '""'
    if os.path.isdir("csv") == False: # if file does not exist, create instead
        os.makedirs("csv")

    unknown_columns = []
    unknown_datatypes = []
    extension = path.split(".")[1]
    if(extension == "csv"): # if file is csv
        df = pd.read_csv(path)
        dataTypes = df.dtypes
        dataColumns = df.columns
        return dataTypes,dataColumns,df,unknown_columns,unknown_datatypes
    
    elif(extension == "txt"): # if file is txt
        dataColumns = []
        values = []
        dataTypes = []
        isTypes = assumption
        unknown_datatypes = set()
        isColumn = True
        with open(path,mode = "r") as file: 
            for line in file:
                # read line and split the columns/types
                line = (line.rstrip('\n')).split(delimitter)
                line = [col.strip(qualifier) for col in line]

                # if value types/columns
                if isTypes:
                    dataTypes = line
                    isTypes = False
                    
                elif isColumn:
                    dataColumns = line
                    isColumn = False

                # if values
                else:
                    values += [line]
        # dataframe is created
        df = pd.DataFrame(data = values, columns = dataColumns)
        if assumption: # assign datatypes to the columns
            df,unknown_datatypes,unknown_columns = assign_datatypes(df,dataTypes) 

        else: # data types are already correct
            dataTypes = df.dtypes

        return dataTypes,dataColumns,df,unknown_datatypes,unknown_columns

#Check if file is valid
def allowed_file(filename):
    """
    Check if the file is valid.
    
    Parameters:
    :filename: -- path to the file.

    Return true if extension is correct.
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def assign_datatypes(data,dataTypes):
    """
    Change the datatypes in the columns of data with respect to the dataTypes
    Parameters:
    :data: -- given dataframe whose dataTypes are all strings
    :dataTypes: -- given list of data types that determines the types of the data
    """
    unknown_datatypes = set()
    unknown_columns = set()
    assigned_datatypes = session.get("assigned_types")
    assigned_columns = session.get("assigned_columns")
    #print("Assigned datatypes = ",assigned_datatypes)
    #print("Assigned columns = ",assigned_columns)

    for i in range(len(dataTypes)):
        if assigned_datatypes.get(dataTypes[i]): # change columns 
            new_type = assigned_datatypes.get(dataTypes[i])
            #print("Changing {} to {}...".format(dataTypes[i],new_type)) 
            dataTypes[i] = new_type
    
    for i in range(len(dataTypes)):
        try:
            if assigned_columns.get(data.columns[i]): # column value couldnt transformed
                new_type = assigned_columns.get(data.columns[i]) 
                #print("Changing {} to {}...".format(data.columns[i],new_type)) 
                data.iloc[:,i] = data.iloc[:,i].astype(new_type)
            else: 
                data.iloc[:,i] = data.iloc[:,i].astype(dataTypes[i]) # try assigning type
        except TypeError: # if type is wrong
            unknown_datatypes.add(dataTypes[i])

        except ValueError: # if type is correct but values can not be converted
            unknown_columns.add(data.columns[i])
    #print("Unknown dtypes : ", unknown_datatypes)
    #print("Unknown columns : ", unknown_columns)
    return data,list(unknown_datatypes),list(unknown_columns)

def choose_attributes(df,attributes):
    """
    Choose the attributes and return the new chosen dataframe.
    Parameters:
    :df: -- given dataframe 
    :attributes: -- features/columns that is picked by the user
    """
    chosen_df = df[attributes]
    unchosen_df = df.drop(attributes,axis=1)
    return chosen_df,unchosen_df

def groupColumns(df):
    """
    Group the columns in DataFrame by their dtypes.
    Parameters
    :df: -- original data with columns
    """
    
    # initilize empty lists
    dtypeArr = []
    columnArr = []
    lens = []
    returnArr = []
    type_dct = {str(k): list(v) for k, v in df.groupby(df.dtypes, axis=1)}
    type_dct = OrderedDict(sorted(type_dct.items(), key=lambda i: -len(i[1])))

    for types in type_dct: # sort column names in the type dictionary
        type_dct[types].sort()
        columnArr.append(type_dct[types])
        dtypeArr.append(types)
        lens.append(len(type_dct[types]))
    
    for i in range(max(lens)): # finalize the result
        arr = []
        for k in range(len(dtypeArr)):
            if(i < lens[k]):
                arr.append(columnArr[k][i])
        returnArr.append(arr)
    return dtypeArr, returnArr

    
def load_(UPLOAD_FOLDER,files):
    """
    Load the file uploaded by user. Check if its valid or not. If valid, return the DataFrame
    Parameters:
    :UPLOAD:FOLDER: -- path to the uploaded file
    :files: -- file dictionary provided by Flask
    """
    from werkzeug.utils import secure_filename
    if 'file' not in files:
        flash('No file part')
        return None
    file = files['file']

    # check if user does not send any file
    if file.filename == '':
        flash('No selected file.')
        return None

    # a valid file is submitted. 
    if file and allowed_file(file.filename):
        # construct the file path
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER,file.filename) 
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        # get delimitter and qualifier form the form.
        delimitter = request.form['delimitter']
        qualifier = request.form['qualifier']

        if request.form.get('is-value-type'):
            assumption = True
        else:
            assumption = False

        # read the file and apply the model
        _, _, df,unknown_types,unknown_columns = load_dataset(file_path,delimitter=delimitter,qualifier = qualifier, assumption=assumption)
        if len(unknown_types) != 0: # some types are unknown
            flash("Some types in provided file is unknown! ({}) Please assign it from existing types!".format(unknown_types))
            session["unknown_types"] = unknown_types
        if len(unknown_columns) != 0:
            flash("Some columns in can not be converted to assigned datatype! ({}) Please assign it from existing types!".format(unknown_columns))
            session["unknown_columns"] = unknown_columns

        return df,unknown_types,unknown_columns

def concat_columns(data,selected_columns):
    """
    Concat object/discrete columns to create new columns.
    :data: -- Dataframe that will be used
    :selected_columns: -- Selected object columns
    """
    return data[selected_columns].apply(lambda x : ','.join(x.dropna().astype(str)),axis=1)



def check_float(potential_float):
    """
    Check if given expression is float / number.
    Parameters:
    :potential_float: -- unknown type 
    """
    if potential_float is None:
        return False
    try:
        float(potential_float)
        return True
    except ValueError:
        return False
def handle_actions(data,actions):
    """
    Handle the actions and clear the empty inputs and check the errorous inputs.
    
    For object variables -- errorous does not exist
    For continious variables, error can be a non-numerical value
    """
    handled_actions = {}
    
    for col in actions.keys():
        action_list = actions[col]
        if data.dtypes[col] == object: # column is object
            handled_actions[col] = action_list
        else:
      
            #First parameter is empty. Fill it with minimum (Lower Boundary)
            if (action_list[0] == "") or (check_float(action_list[0]) == False):
                action_list[0] = data[col].min()
                
            else:
                action_list[0] = float(action_list[0])
            
            #Second parameter is empty. Fill it with maximum (Upper Boundary)
            if (action_list[1] == "") or (check_float(action_list[1]) == False):
                action_list[1] = data[col].max()
                
            else:
                action_list[1] = float(action_list[1])
                
            
            handled_actions[col] = action_list
    return handled_actions
    

def filter_data(data, actions):
    """
    Filter the dataset with given boundaries
    :data: -- is the dataframe
    :actions: -- is the list of parameters and the actions that should be taken. 
    """
    copy_data = data.copy()
    handled_actions = handle_actions(data,actions)

    for column, action in handled_actions.items():
        if data.dtypes[column] == object: #Datatype is object/discreate
            condition = data[column].isin(action)
            
        else:
            condition = (data[column] >= action[0]) & (data[column] < action[1])
        data = data[condition]
    return copy_data.loc[data.index]

def remove_temp_files(user_id, head = "temp"):
    """
    Remove temporary files. This function will be used when the user is log out.
    Parameters:
    :user_id: -- id of the user
    :head: -- folder that will be deleted
    """
    if os.path.isdir(head) == False: # if file does not exist, create instead
        os.makedirs(head)
    if(user_id == None):
        flash("An error occured while removing files.")
    arr = os.listdir(head)
    user_files = []
    for file in arr: # split the file into an array
        print(file.split("-"))
        if file.split("-")[0] == str(user_id): # file belongs to user that just logged out
            user_files += [file]
    
    for file in user_files: # remove all files belongs to the user
        path_temp = os.path.join(head,file)
        is_directory = os.path.isdir(path_temp)
        if not is_directory:
            os.remove(path_temp)
        else:
            try: # if path is folder, remove 
                shutil.rmtree(path_temp)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


def load_temp_dataframe(user_id, body = "-df-temp",method = "feather",head = "temp"):
    """
    Load temporary dataframe into temp folder for user. 
    Everytime we need dataframe, reload it.
    :user_id: -- is the id of the user 
    :method: -- how to save , default valeus is feather as it does great job at saving/loading temporary files
    """
    if os.path.isdir(head) == False: # if file does not exist, create instead
        os.makedirs(head)

    if method == "feather":
        extension = ".feather"
    path = os.path.join(head,str(user_id) + body + extension)
    try:
        data = pd.read_feather(path)
    except:
        save_temp_dataframe(pd.DataFrame(),user_id)
        data = pd.read_feather(path)
    data = data.set_index("index") if "index" in data.columns else data
    return data

def save_temp_dataframe(data,user_id, body="-df-temp", method = "feather", head = "temp"):
    """
    Save temporary dataframe into temp folder for user
    Everytime we change dataframe, change it
    :data: -- dataframe that is changed
    :user_id: -- is the id of the user 
    :method: -- how to save , default valeus is feather as it does great job at saving/loading temporary files
    """
    if os.path.isdir(head) == False: # if file does not exist, create instead
        os.makedirs(head)

    
    if method == "feather":
        extension = ".feather"
        path = os.path.join(head,str(user_id) + body + extension)
        data = data.reset_index() # convert index to index column
        data.to_feather(path)
        
    if method == "csv":
        extension = ".csv"
        path = os.path.join(head,str(user_id) + body + extension)
        data.to_csv(path_or_buf = path) # convert index to csv


def save_user_model(model,user_id,body = "-model", method = "pickle", head = "models"):
    """
    Save users trained model.
    Parameters:
    :body: -- body of the filename
    :method: -- method for saving, only pickle exist for now
    :head: -- folder for the file.
    """
    if os.path.isdir(head) == False: # if file does not exist, create instead
        os.makedirs(head)
    if method == "pickle":
        filename = os.path.join(head,str(user_id) + body +  ".sav")
        pickle.dump(model, open(filename, 'wb'))
    if method == "model":
        filename = os.path.join(head,str(user_id) + body)
        model.save(filename)

def load_user_model(user_id,body = "-model", method = "pickle",head = "models"):
    """
    Load users trained model.
    Parameters:
    :body: -- body of the filename
    :method: -- method for saving, only pickle exist for now
    :head: -- folder for the file.
    """
    if os.path.isdir(head) == False: # if file does not exist, create instead
        os.makedirs(head)
    if method == "pickle":
        filename =os.path.join(head,str(user_id) + body + ".sav")
        print(filename)
        return pickle.load(open(filename, 'rb'))
    if method == "model":
        filename = os.path.join(head,str(user_id) + body)
        return tf.keras.models.load_model(filename)

def type_divider(df):
    """
    This function takes a dataframe and creates a dict where each key represents type and value of that type
    is array of columns. 

    Input:
    :df: -- Input dataframe 

    Return:
    :type_dict: -- Divided type-columns dictionary
    """
    unique_dtypes = np.unique(df.dtypes.values)
    type_dict = {}
    for dtype in unique_dtypes:
        columns = df.select_dtypes(include = [dtype]).columns
        type_dict[dtype] = columns.values
    return type_dict

def instance_divider(df):
    """
    This function takes a dataframe and outputs a dict with number of columns assigned to each type.
    """
    no_of_integer = df.select_dtypes(include = [np.int8,np.int16,np.int32,np.int64]).shape[1]
    no_of_inexact = df.select_dtypes(include = [np.float16,np.float32,np.float64]).shape[1]
    no_of_object = df.select_dtypes(include = ["object"]).shape[1]
    other_columns = df.shape[1] - (no_of_integer + no_of_inexact + no_of_object)
    return no_of_integer,no_of_inexact,no_of_object,other_columns

def model_chooser(df,selected_y):
    """
    Determine the task of the model by selected parameters.
    Parameters:
    :df: -- selected data
    :selected_y: -- selected parameters that will be predicted
    """
    no_of_integer,no_of_inexact,no_of_object,other_columns = instance_divider(df[selected_y])
    if other_columns != 0:
        flash("A variable y with no possible model selection has found!")
        return redirect(url_for("select_y"))
    else:
        if no_of_integer != 0: 
            if no_of_inexact == 0  and no_of_object != 0: #All columns are integer or object, use classification
                classification_model = True
                regression_model = False

            elif no_of_object == 0 and no_of_inexact != 0: # All columns are integer or float, use regression
                regression_model = True 
                classification_model = False
            
            elif no_of_inexact != 0 and no_of_object != 0: #Columns are mixed, show error
                flash("No possible model can be selected! Please use different target variables for prediction")
                return False,False

            else: #All columns are integer. Both of regression and classification can be used
                regression_model = True
                classification_model = True
        else: 
            if no_of_inexact == 0  and no_of_object != 0: #All columns are object, use classification
                classification_model = True
                regression_model = False

            elif no_of_object == 0: # All columns are flaot, use regression
                regression_model = True 
                classification_model = False
    return regression_model,classification_model

def remove_empty_lists(data):
    """
    Return None if all elements are empty string.
    """
    data = np.array(data)
    if np.all(data == ""):
        return None
    return data.tolist()

def check_suitable(test_data, new_data):
    """
    Check if test_data used in training and new_data used in prediction is suitable.
    Two dataframe is suitable if number of columns and their data-types are same. If column names are not equal
    flash a warning to user.
    """
    # check if columns match
    if(test_data.shape[1] != new_data.shape[1]): # not have equal number of columns
        if test_data.columns.isin(new_data.columns).all(): # columns match so suitable
            flash("Newly uploaded data has extra columns. Only columns used in model training will be used.")
            return True
        else:
            flash("Data is not suitible with newly uploaded data.")
            return False

    return test_data.columns.isin(new_data.columns).all() # return true if columns match or false if not
    
def assign_default_value(potential_variable,default_value,check_type = "float", negative_allowed = False):
    """
    Check and assign value if the value can not be converted into a float
    """ 
    if check_type == "float": # value can be converted into float
        can_converted = check_float(potential_variable)
        potential_return = default_value if can_converted == False else float(potential_variable)
        
    elif check_type == "int": # value can be converted into int
        can_converted = check_float(potential_variable)
        potential_return = default_value if can_converted == False else int(potential_variable)
        
    elif check_type == "string":
        return default_value if potential_variable == "" else potential_variable

    if negative_allowed: # if negative allowed return variable
        return potential_return

    else: # negative not allowed
        if potential_return >= 0:
            return potential_return
        else:
            return default_value
