import numpy as np
import pandas as pd
import time
from run import ALLOWED_EXTENSIONS
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField, SubmitField,BooleanField
from wtforms.validators import DataRequired,Length,Email,EqualTo

def load_dataset(path,delimitter = ",",qualifier = '"',assumption = False):
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
    extension = path.split(".")[1]
    if(extension == "csv"):
        df = pd.DataFrame(data=file_path, index_col = 0)
        return df
    elif(extension == "txt"):
        dataColumns = []
        values = []
        dataTypes = []
        isTypes = assumption
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
        df.set_index(df.columns[0])
        df.drop([df.columns[0]],inplace=True,axis=1)
        if assumption:
            df = assign_datatypes(df,dataTypes)
        else:
            dataTypes = df.dtypes
        return dataTypes,dataColumns,df
    print("Error")

#Check if file is valid
def allowed_file(filename):
    """
    Check if the file is valid.
    
    Parameters:
    :filename: -- path to the file.

    Return true if extension is correct.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def assign_datatypes(data,dataTypes):
    """
    Change the datatypes in the columns of data with respect to the dataTypes
    Parameters:
    :data: -- given dataframe whose dataTypes are all strings
    :dataTypes: -- given list of data types that determines the types of the data
    """
    types = {
        "INT" : np.int32,
        "FLOAT" : np.float32,
        "CATEGORICAL" : np.object,
        "BOOLEAN" : np.bool
    }

    for i in range(len(dataTypes)):
        data.iloc[:,i].astype(types(dataTypes[i]))


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
    

class StringForm(FlaskForm):
    delimetter = StringField('Username', validators = [DataRequired()])
    submit = SubmitField("Sign Up")
