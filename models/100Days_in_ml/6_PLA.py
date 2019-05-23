# -*- coding: utf-8 -*-
# PLA算法

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../../dataset/100_in_ml/data1.csv', header=None)

# 样本输入，维度（100，2）
X = data.iloc[:,:2].values
# 样本输出，维度（100，）
y = data.iloc[:,2].values

plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='Positive') #前50
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='Negative') #后50
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.title('Original Data')
plt.show()

# 均值
u = np.mean(X, axis=0)
# 方差
v = np.std(X, axis=0)
X = (X - u) / v # 注意

# 作图
plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='Negative')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.title('Normalization data')
plt.show()

# 直线初始化
# X加上偏置项
X = np.hstack((np.ones((X.shape[0],1)), X))
# 权重初始化
w = np.random.randn(3,1)

for i in range(100):
    s = np.dot(X, w) # innerProduct = sum(w * x) 计算内积
    y_pred = np.ones_like(y)
    # 接下来的 if 语句判断如果内积小于等于 0，则是负例，否则是正例
    loc_n = np.where(s < 0)[0]
    y_pred[loc_n] = -1
    num_fault = len(np.where(y != y_pred)[0])
    print('第%2d次更新，分类错误的点个数：%2d' % (i, num_fault))
    if num_fault == 0:
        break
    else:
        t = np.where(y != y_pred)[0][0]
        w += y[t] * X[t, :].reshape((3,1))

# numpy 提供的 append 函数可以扩充一维数组，可以自己实验一下。

# 直线第一个坐标（x1，y1）
x1 = -2
y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
# 直线第二个坐标（x2，y2）
x2 = 2
y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
# 两点确定一条直线

# 作图
plt.scatter(X[:50, 1], X[:50, 2], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 1], X[50:, 2], color='red', marker='x', label='Negative')

plt.plot([x1,x2], [y1,y2],'r')
#感知机可以是线，圈平面，空间
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.show()