# -*- coding: utf-8 -*-
import numpy as np

#用一个全局变量，来保存最优的划分数据
bestFeatures=[]
bestValues=[]

def loadDataSet(fileName):
    dataMat = []
    fileRead = open(fileName)
    for line in fileRead.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float()
        floatLine = map(float, curLine)
        dataMat.append(floatLine)
    return np.mat(dataMat)


myData = loadDataSet('../../dataset/100_in_ml/ex00.txt')
print(myData.shape)

import matplotlib.pyplot as plot

# np.matrix.A--->matrix转化为array
plot.scatter(myData[:, 0].A, myData[:, 1].A, color="blue")
plot.xlabel("x")
plot.ylabel("y")
plot.show()


# 二元切分法,x>a和x<=a,其中的a属于某个特征xi的所有取值集合
# notice:切分的特征并没有删除，这与前面ID3算法的决策树分类中，切分数据不同!!!
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :]
    return mat0, mat1


mat0, mat1 = binSplitDataSet(myData, 0, 0.5)
print(mat0.shape, mat1.shape)


# 树回归的叶子节点，数据集合中v的均值
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 用y的总均方差代表连续型数据的混乱度
def regErr(dataSet):
    # 总均方差=均方差*num
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 预剪枝与后剪枝的比较
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]  # 均方差变化的最小幅度，用于预剪枝
    tolN = ops[1]  # 划分的数据集的最小size

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit condition 1:
        # 如果所有的y的值都相等，这里用set去重,直接返回叶子节点，不做划分
        return None, leafType(dataSet)  # 直接返回
    m, n = np.shape(dataSet)  # 数据集合大小
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)  # 原来数据集合的总均方差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featureIndex in range(n - 1):  # 遍历每一个特征：因为下标-1是y的值，不是特征
        for splitVal in set(dataSet[:, featureIndex].A.ravel()):  # 遍历当前特征的每一个取值
            mat0, mat1 = binSplitDataSet(dataSet, featureIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                # 如果划分之后的数据集合size太小了，直接跳过，不划分
                continue
            newS = errType(mat0) + errType(mat1)  # 划分之后的两个小数据集合的总均方差
            if newS < bestS:  # 选出最优值
                bestIndex = featureIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold,don't do the split
    if (S - bestS) < tolS:
        # 误差变化的幅度太小，直接返回叶子节点，不做划分
        return None, leafType(dataSet)  # exit condition 2 #退出条件2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 最优的划分子集合
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit condition 3
        # 划分的子集合太小,直接返回叶子节点，不做划分
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


#递归创建回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # assume dataSet is NumPy Mat so we can array filtering
    feature, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果返回的是没有划分，和叶子节点的值
    if feature == None:
        # if the splitting hit a stop condition return val
        return val # bestValue
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = val

    bestFeatures.append(feature)
    bestValues.append(val)

    lSet, rSet = binSplitDataSet(dataSet, feature, val)  # 分成左右两个子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree

# myData = loadDataSet('../../dataset/100_in_ml/ex00.txt')
# myTree = createTree(myData)
# print(myTree)


import matplotlib.pyplot as plt
plt.scatter(myData[:,0].A,myData[:,1].A,color="blue")

y = myData[:,-1].A
y_min = np.min(y) - 0.1
y_max = np.max(y) + 0.1
y_val = np.arange(y_min,y_max,0.01)
for i in xrange(len(bestValues)):
    plt.plot([bestValues[i]]*y_val.shape[0],y_val,color="black") #画出那条分界线
plt.xlabel("x")
plt.ylabel("y")
plt.show()