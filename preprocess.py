import numpy as np  
import pandas as pd
from sklearn import preprocessing
def stringDecoder(y, arrEncoder, selectedY):
    decodedY = []
    
    for i in range(len(selectedY)):
        if(len(selectedY) > 1):
            for column, encoder in arrEncoder:
                if column == selectedY[i]:
                    decodedY.append(encoder.inverse_transform(y[i]))
        else:
            for column, encoder in arrEncoder:
                if column == selectedY[i]:
                    decodedY = encoder.inverse_transform(y)
    return decodedY

def stringEncoder(df, columns):
	"""
	Encodes the string values 0 to n-1
	returns
	"""
	arrEncoder = []
	
	for column in columns:
		encoder = preprocessing.LabelEncoder()
		df[column] = encoder.fit_transform(df[column])
		arrEncoder.append((column, encoder))
	
	return df, arrEncoder

def dropNanAndDuplicates(df, notNanPercantage = 1):
    threshval = int(len(df) * notNanPercantage)
    df = df.dropna(thresh= threshval, axis=1)
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    return df

def scale(df, selectedY):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scalerY = preprocessing.MinMaxScaler(feature_range=(0,1))
    scalerY.fit(df.loc[:,selectedY])
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled, columns= df.columns, index= df.index)
    return df, scaler, scalerY

"""
def shuffle_list(x, y, seed = None):
    #Shuffle the list and drop the None type values
    if seed != None:
        np.random.seed(seed)
    
    dynamic_list = []
    
    for i  in range(x.shape[0]):
    
        if type(x[i]) == type(None):
            continue
        
        dynamic_list.append(i)
    
    np.random.shuffle(dynamic_list)
    return x[dynamic_list], y[dynamic_list]
    
def clear_NONE_TYPE(x, y):
    #Clear Nontypes in array
    dynamic_list = []
    
    for  i in range(x.shape[0]):
        
        if type(x[i]) == type(None):
            continue
        
        dynamic_list.append(i)
        
    return x[dynamic_list], y[dynamic_list]
    
def fetch_ID_dataframe_last(df, min_observation = 2, seperate = False):
    ID_SET = set(df['ID'])
    ID_LIST = list(ID_SET)
    
    leftArray = np.emty(len(ID_LIST) *2, dtype = object)
    leftPredict = np.zeros(shape=len(ID_LIST)*2, dtype= object)
    
    rightArray = np.emty(len(ID_LIST) *2, dtype = object)
    rightPredict = np.zeros(shape=len(ID_LIST)*2, dtype= object)
    
    resultArray = np.emty(len(ID_LIST) *2, dtype = object)
    predictArray = np.zeros(shape=len(ID_LIST)*2, dtype= object)
    
    #Placeholders
    placeholder = 0
    placeholder_left = 0
    placeholder_right = 0
    
    for i in ID_LIST: #For every index in dataframe
    
        personData = df[df['ID'] == i]
    
        #Sort the values and drop duplicates (get one observation for each date-time)
        personData = personData.sort_values(by = ['Date.Time'])
    
        #Get the left and right eyes data seperetly
        personData_left = personData[personData['Eye'] == 'OS(Left)']
        personData_left = personData_left.drop_duplicates(subset = ['Date.Time'])
        
        personData_right = personData[personData['Eye'] == 'OD(Right)']
        personData_right = personData_right.drop_duplicates(subset = ['Date.Time'])
        
        for personData in [personData_left, personData_right]:
        
            #If eye has less than min_observation, continue
            if(personData.shape[0] < min_observation):
                continue
            
            isLeft = (personData['Eye'] == 'OS(Left)').any()
            personData.drop(['ID', 'Eye', 'idEye'], axis=1)
            
            #Get the ESI values and drop the unnecessary columns
            personData_y = personData['ESI']
            personData = personData.drop(['ESI'], axis=1)
            
            #Get the timesize and add sample to the empty array
            time_size = personData.shape[0]
            lastData = personData_y
            
            if seperate:
                if isLeft:
                    leftArray [placeholder_left] = personData
                    leftPredict[placeholder_left] = lastData
                    placeholder_left += 1
                else:
                    rightArray[placeholder_right] = personData
                    rightPredict[placeholder_right] = lastData
                    placeholder_right += 1
                
            resultArray[placeholder] = personData
            predictArray[placeholder] = lastData
            placeholder += 1
                
    big_x, big_y = clear_NONE_TYPE(resultArray, predictArray)
    if seperate:
        left_x, left_y = clear_NONE_TYPE(leftArray, leftPredict)
        right_x, right_y = clear_NONE_TYPE(rightArray, rightPredict)
        return ((big_x, big_y),(left_x, left_y),(right_x, right_y))
        
    return big_x, big_y
"""