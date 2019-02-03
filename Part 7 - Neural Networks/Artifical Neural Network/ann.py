#Artificial Neural Network

# 1. Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
X=oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising ANN
classifier = Sequential()

# Add one input layer and one hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))

# Add second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training set
classifier.fit(X_train,y_train, batch_size=10, nb_epoch=100)

# Making the predictions and evaluating the model

# Predict the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

