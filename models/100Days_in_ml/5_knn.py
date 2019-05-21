# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

dataset = pd.read_csv('../../dataset/100_in_ml/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = Y, cv = 10)

mean_accuracy = accuracies.mean()
# 偏差，之前那本书有说
standard_deviation = accuracies.std()

# Visualise Results - code taken from www.superdatascience.com/machine-learning

from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop= x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop= x_set[:, 1].max() + 1, step=0.01))

plot.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plot.xlim(x1.min(), x1.max())
plot.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                 c = ListedColormap(('red', 'blue'))(i), label = j)


plot.title('K-NN Classification (Test set)')
plot.xlabel('Age(scaled)')
plot.ylabel('Estimated Salary(scaled)')
plot.legend()
plot.show()









# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_train, y_pred)

