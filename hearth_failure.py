#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:26:03 2020

@author: catarinareis
"""

import pandas as pd
import tensorflow
from tensorflow import keras
from keras import layers
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from sklearn.ensemble import IsolationForest

pd.set_option('display.max_columns', 30)
df = pd.read_csv("heart_failure_dataset.csv")
df.head()
df.isnull().sum()
df.describe()

plt.figure(figsize=[20,10])
sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)

# Getting only highly correlated features
Features = ['time', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'age']

# Setting number of training samples
training_samples = 250
data = df[Features]
labels = df.iloc[:, -1]

# Applying scaling to data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Splitting data to train and test
X_train = data[:training_samples]
y_train = labels[:training_samples]
X_test = data[training_samples:]
y_test = labels[training_samples:]

# Box plot before dealing with outliers
test=px.box(X_train, points='all')
test.show()
