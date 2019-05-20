# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

dataset = pd.read_csv('../../dataset/100_in_ml/studentscores.csv')

X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression = regression.fit(X_train, Y_train)

# step predict
# Y_test_pred = regression.predict(X_test)
Y_train_pred = regression.predict(X_train)

plot.scatter(X_train, Y_train, color='red')
plot.plot(X_train, Y_train_pred, color='blue')
plot.show()