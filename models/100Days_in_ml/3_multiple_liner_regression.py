# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

dataset = pd.read_csv('../../dataset/100_in_ml/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
onehotEncoder = OneHotEncoder(categorical_features=[3])
X = onehotEncoder.fit_transform(X)#.toarray()
print(X)

### Avoiding Dummy Variable Trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
Y_pred = regression.predict(X_train)
plot.scatter(X_train, Y_train, color='red')
plot.plot(X_train, Y_pred, color='blue')
plot.show()