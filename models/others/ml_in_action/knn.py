import numpy as np
import matplotlib.pyplot as plt

def load_data(fileName):
    dataSet = []
    label = []
    file = open(fileName)
    for line in file.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append(lineArr[0:3]) # 这个是开区间的！
        label.append(lineArr[3:])
    return np.array(dataSet, dtype=np.float64), np.array(label, dtype=np.int)


def plot(x, y):
    label1 = np.where(y.ravel() == 1)
    plt.scatter(x[label1, 0], x[label1, 1], marker='x', color='r', label='dis like')
    plt.xlabel('pilot distance')
    plt.ylabel('game time')
    plt.legend(loc='upper left')
    plt.show()

data, label = load_data('datingTestSet2.txt')
print(data.shape)
print(label.shape)

plot(data, label)