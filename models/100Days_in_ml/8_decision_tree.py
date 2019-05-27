# -*- coding:utf-8 -*-

import math
import operator
from matplotlib import pyplot as plot
import pickle


def createDataSet():
    # dataSet = [[1, 1, 'yes'],
    #            [1, 1, 'yes'],
    #            [1, 0, 'no'],
    #            [0, 1, 'no'],
    #            [0, 1, 'no']]
    # featureNames = ['no surfacing', 'flippers']  # 不浮出水面是否存活 ，有无脚蹼
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    featureNames = ['age', 'has job', 'has estate', 'loan condition']  # 特征标签
    # change to discrete values
    return dataSet, featureNames


"""
#返回经验熵(香农熵)
# 计算信息熵,因为我们会利用最大信息增益的方法划分数据集-----
# 看哪个特征划分使得，信息熵(数据无序度)减小的最多
"""


def entropy(dataSet):
    num = len(dataSet)
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 统计每个类别的数量
        currentlabel = featVec[-1]  # pick the last element as key
        if currentlabel not in labelCounts.keys():  # 当前标签
            labelCounts[currentlabel] = 0  # default value as 0
        labelCounts[currentlabel] += 1  # count

    print('labelcount is ', labelCounts)

    entropy = 0.0
    for key in labelCounts:
        probability = float(labelCounts[key]) / num  # 当前key的概率
        entropy -= probability * math.log(probability, 2)  # log base on 2
    return entropy


myData, myFeaturesNames = createDataSet()
print('old dataset is ', myData)

entropy1 = entropy(myData)
print('my test entropy should be 0.97095 :', entropy1)

# 添加一类，mabey,yes,no   ====熵越高，表明数据越混乱
myData[0][-1] = "mabey"
print('new dataset is ', myData)


# 按照给定的特征axis,根据他的取值value，划分数据集，返回新的数据集合，少了1个特征（划分依据的那个特征axis）
def splitDataSet(dataSet, axis, value):
    """
        # 去掉axis特征
        #将符合条件的添加到返回的数据集
        :param dataSet:
        :param axis:
        :param value:
        :return:
    """
    finalDataSet = []
    for dataVec in dataSet:
        if dataVec[axis] == value:
            tempVec = dataVec[:axis]  # 0--(axis-1)
            tempVec.extend(dataVec[axis + 1:])  # (axis+1)--(-1) #所以减去了axis
            finalDataSet.append(tempVec)
    return finalDataSet


myData, myFeaturesNames = createDataSet()
print splitDataSet(myData, 0, 1)  # axis = 0,且这个特征的值=1


# 看哪个特征划分使得，信息熵(数据无序度)减小的最多
def chooseBestFeature2Split(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后1列是类别
    baseEntropy = entropy(dataSet)  # 首先计算原始的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate所有特征
        # 将数据集中的第i个特征的值，放到一个list中
        featureList = [example[i] for example in dataSet]  # 字典表达式就这样写
        uniqueValue = set(featureList)
        newEntropy = 0.0
        for value in uniqueValue:
            # 对第i个特征，针对某个值划分
            subDateSet = splitDataSet(dataSet, i, value)
            probability = len(subDateSet) / float(len(dataSet))  # 用原来的dataset去除，计算子集的概率
            newEntropy += probability * entropy(subDateSet)  # 累加小dataset的熵
        tempInfoGain = baseEntropy - newEntropy
        if (tempInfoGain > bestInfoGain):
            bestInfoGain = tempInfoGain  # replase gain
            bestFeature = i  # 标注下标
    return bestFeature  # 信息增益最大的(最优)特征的索引值


# test
index = chooseBestFeature2Split(dataSet=myData)
print "best feature shoule be 0: ", index


# 对字典排序，取得最大值:将多数的类别标签作为“叶子节点”的类别
def majoriryCnt(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            # 根据字典的值降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majoriryCnt(classList)
    bestFeat = chooseBestFeature2Split(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)  # 最优特征的标签
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree


def getNumLeafs(myTree):
    # initialize leafs
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,
    # 所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def createPlot(inTree):
    fig = plot.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plot.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '\u54c8\u54c8')
    plot.show()


import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# def plotNode(nodeTxt, centerPt, parentPt, nodeType):
#     #font = mpl.font_manager.FontProperties(fname='SimHei.ttf')
#     font = FontProperties(fname='/Library/Fonts/SimSun.ttf', size=10)
#     arrow_args = dict(arrowstype="<-")
#     # 绘制结点
#     createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
#                             xytext=centerPt, textcoords='axes fraction',
#                             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname='/Library/Fonts/SimSun.ttf', size=10)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotMidText(cntrPlot, parentPlot, txtString):
    xMid = (parentPlot[0] - cntrPlot[0]) / 2.0 + cntrPlot[0]
    yMid = (parentPlot[1] - cntrPlot[1]) / 2.0 + cntrPlot[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPlt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    centerPlt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(centerPlt, parentPlt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, centerPlt, parentPlt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], centerPlt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPlt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPlt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def plotTree2(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

"""
使用决策树分类
Parameters:
	inputTree - 已经生成的决策树
	featLabels - 存储选择的最优特征标签
	testVec - 测试数据列表，顺序对应最优特征标签
"""


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    createPlot(myTree)
    testVec = [0, 1]
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')
