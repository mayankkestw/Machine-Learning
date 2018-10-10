#Multiple Linear Regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Read data
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,3]=labelencoder_X.fit_transform(x[:,3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
x=oneHotEncoder.fit_transform(x).toarray()

#Avoiding Dummy variable trap
x = x[:,1:]

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=1/3, random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''