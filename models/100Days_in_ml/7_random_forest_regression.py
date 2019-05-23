# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Read Data
from sklearn.datasets import load_boston 
dataset = load_boston()

# Choose which features to use
x = dataset["data"][:, [7, 9]] # using DIS and TAX features
y = dataset["target"]     # output value

# Split data into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x[:,:], y, test_size = 0.2, random_state = 42)

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
sc_x    = StandardScaler()
x_train = sc_x.fit_transform(x_train) # Scaling the data
x_test  = sc_x.transform(x_test)

# Train Model
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators = 10, random_state = 42)
regr.fit(x_train, y_train)

# Predict Results
y_pred = regr.predict(x_test)

# Measure Accuracy
from sklearn.metrics import mean_squared_error
acc = mean_squared_error(y_test, y_pred)

# 直线初始化
# X加上偏置项
X = np.hstack((np.ones((x.shape[0],1)), x))
# 权重初始化
w = np.random.randn(3,1)

# for i in range(100):
#     s = np.dot(X, w) # innerProduct = sum(w * x) 计算内积
#     y_pred = np.ones_like(y)
#     # 接下来的 if 语句判断如果内积小于等于 0，则是负例，否则是正例
#     loc_n = np.where(s < 0)[0]
#     y_pred[loc_n] = -1
#     num_fault = len(np.where(y != y_pred)[0])
#     print('第%2d次更新，分类错误的点个数：%2d' % (i, num_fault))
#     if num_fault == 0:
#         break
#     else:
#         t = np.where(y != y_pred)[0][0]
#         w += y[t] * X[t, :].reshape((3,1))

# numpy 提供的 append 函数可以扩充一维数组，可以自己实验一下。

# 直线第一个坐标（x1，y1）
x1 = -2
y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
# 直线第二个坐标（x2，y2）
x2 = 2
y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
# 两点确定一条直线



# Visualise Results
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('DIS values')
ax.set_ylabel('TAX values')
ax.set_zlabel('House Price(1k$)')
x_test = sc_x.inverse_transform(x_test)
ax.scatter(x_test[:,0], x_test[:,1], y_test, color = 'r')
ax.scatter(x_test[:,0], x_test[:,1], y_pred, color = 'b')

plt.plot([x1,x2], [y1,y2],'r')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.show()