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