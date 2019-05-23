# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

dataset = pd.read_csv('../../dataset/100_in_ml/Social_Network_Ads.csv')


X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values # special column

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)

#feature划分？
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = X_train.astype(float)
X_test = X_test.astype(float)
X_train = sc.fit(X_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)
