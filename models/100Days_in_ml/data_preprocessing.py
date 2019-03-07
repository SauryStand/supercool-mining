# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('dataset/Data.csv')
X = dataset.iloc[:, :-1].value  # pick up target column
Y = dataset.iloc[:, 3].values

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])  # this is mean to choose line 1-3?
