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
