# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder
import matplotlib.pyplot as plot

dataset = pd.read_csv('../../dataset/Data.csv')
X = dataset.iloc[:, :-1].values  # pick up target column
Y = dataset.iloc[:, 3].values

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3]) # 不包含第三列
X[:, 1:3] = imputer.transform(X[:, 1:3])  # this is mean to choose line 1-3?


## Step 3: Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) # 1 is get the average count, seems like
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#step4 encoding categorial data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])


onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# plot.scatter(X_train, X_train, color='red')
# plot.show()


