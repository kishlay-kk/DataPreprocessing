# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:09:52 2018

@author: kishl
"""

#DATA PREPROCESSING

# STEP 1:----Import Libraries
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plot

# STEP 2:----Import the dataset
data=pd.read_csv('Data.csv')

# STEP 3:----Separate Dependent and Independent Varriables
x=data.iloc[:,0:3]
y=data.iloc[:,3]

#STEP 4:----Check and Replace Empty Values if any
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="mean",axis=0)
x.iloc[:,1:3]=imp.fit_transform(x.iloc[:,1:3]) 

#STEP 5:----Encode the non-int non-float values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label= LabelEncoder()
x.iloc[:,0]=label.fit_transform(x.iloc[:,0])
onehot= OneHotEncoder(categorical_features= [0])
x= onehot.fit_transform(x).toarray()

label2= LabelEncoder()
y=label2.fit_transform(y)

#STEP 6:----Splitting into Training Set and Test Set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

#STEP 7:----Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)