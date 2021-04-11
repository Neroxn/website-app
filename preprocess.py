import numpy as np  
import pandas as pd
from sklearn import preprocessing

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

def dropNanAndDuplicates(df, notNanPercantage = 1)
    threshval = int(len(df) * notNanPercantage)
    df = df.dropna(thresh= threshval, axis=1)
    df = df.drop_duplicates()
    return df

def scale(df)
    scaler = MinMaxScale(feature_range=(0,1))
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled, columns= df.columns, index= df.index)
    return df, scaler