from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
from math import log
from numpy import inf

# 获得数据集的
def getDataSet():
    dataSet = load_breast_cancer()
    # 获得数据集的数据
    data = dataSet.data
    target = dataSet.target
    # 获得数据集的label
    label = dataSet.feature_names

    # 将标签加入到数据集的最后一列
    data = np.c_[data, target]
    print("数据导入完毕...")

    print('The number of datas: ' + str(len(data)))
    print('The number of targets: ' + str(len(target)))
    return data, label


# 计算信息熵
def calcShannonEnt(dataSet):
    # 获取数据集中数据的条数
    numEntries = len(dataSet)
    labelCounts = {}

    # 对数据进行遍历
    for featVec in dataSet:
        # 获取数据的标签，即最终的类别
        currentLabel = featVec[-1]
        # 如果字典中没有此标签，即加入字典中。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 若有，则++
        labelCounts[currentLabel] += 1
    # 初始化信息熵
    shannonEnt = 0.0
    # 遍历标签字典
    for key in labelCounts:
        # 计算每一个类别出现的概率（类别出现的次数/数据总数）
        prob = float(labelCounts[key])/numEntries
        # 获得信息熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#统计classList中出现此处最多的元素(类标签)
def majorityCnt(classList):
    import operator
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素

# 划分数据集, axis:按第几个特征划分, value:划分特征的值, LorR: value值左侧（小于）或右侧（大于）的数据集
def splitDataSet_c(dataSet, axis, value, LorR='L'):
    # 创建分类列表
    retDataSet = []
    # 这行不写也行
    featVec = []

    # 如果要划分的是小于此值的特征
    if LorR == 'L':
        # 遍历每个数据样本
        for featVec in dataSet:
            # 如果样本中axis对应的特征的值小于给定值
            if float(featVec[axis]) < value:
                # 就将此样本放入返回数据集中
                retDataSet.append(featVec)
    # 如果要划分的是大于此值的特征
    else:
        for featVec in dataSet:
            if float(featVec[axis]) > value:
                retDataSet.append(featVec)
    return retDataSet


# 选择最好的数据集划分方式
# labelproperty 是特征的类型 0：离散值 1：连续值
def chooseBestFeatureToSplit_c(dataSet, labelProperty):
    # 获得特征的个数
    numFeatures = len(labelProperty)
    
    # 计算当前数据集的信息熵 
    baseEntropy = calcShannonEnt(dataSet)  # 计算根节点的信息熵
    # 初始化最好的信息增益值，和最好分割的特征序号
    bestInfoGain = 0.0
    bestFeature = -1
    # 连续的特征值，即最佳划分值
    bestPartValue = None 

    # 遍历每个特征
    for i in range(numFeatures): 

        # 这句话的逻辑是：从数据集中先提取每个样本，再将样本的固定特征值放入列表中
        featList = [example[i] for example in dataSet]
        # 去除特征值的冗余项
        uniqueVals = set(featList)
        # 初始化特征信息熵 
        newEntropy = 0.0
        # 初始化，最佳划分连续特征值
        bestPartValuei = None

        # 如果是离散的特征
        if labelProperty[i] == 0:  # 对离散的特征
            for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
                subDataSet = splitDataSet_c(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
       
       
        # 如果是连续的特征
        else: 
            # 先将特征进行排序
            sortedUniqueVals = list(uniqueVals) 
            sortedUniqueVals.sort()

            # 初始化划分列表
            listPartition = []
            # 初始化最小的信息熵
            minEntropy = inf

            # 遍历需要划分的结点数
            # 需要遍历的结点数，是特征值的个数-1。因为取得是每两个特征值的平均值
            for j in range(len(sortedUniqueVals) - 1):
                # 获取划分值
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2

                # 获取此划分点左、右侧的子数据集
                dataSetLeft = splitDataSet_c(dataSet, i, partValue, 'L')
                dataSetRight = splitDataSet_c(dataSet, i, partValue, 'R')
                # 计算此划分点左、右两侧的出现概率
                probLeft = len(dataSetLeft) / float(len(dataSet))
                probRight = len(dataSetRight) / float(len(dataSet))
                # 获取当前划分点的信息熵
                Entropy = probLeft * calcShannonEnt(dataSetLeft) + probRight * calcShannonEnt(dataSetRight)
                # 获取最小的信息熵
                if Entropy < minEntropy: 
                    minEntropy = Entropy
                    # 记录最优划分值
                    bestPartValuei = partValue
            newEntropy = minEntropy
        # 计算信息增益
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        # 取最大的信息增益对应的特征
        if infoGain > bestInfoGain:  
            bestInfoGain = infoGain
            bestFeature = i
            bestPartValue = bestPartValuei
    # 返回最好的特征值和最好的划分值
    return bestFeature, bestPartValue




# 创建树
def createTree_c(dataSet, labels, labelProperty):
    import copy
    # 获取每一个样本的标签
    classList = [example[-1] for example in dataSet]
    
    # 有两种结束方式
    # 当叶子节点中只剩下一种类别的时候，返回标签list
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 如果没有特征继续分割，就返回投票值 
    if len(dataSet[0]) == 1:  
        return majorityCnt(classList)

    # 获得最优的特征索引，和最佳分割值
    bestFeat, bestPartValue = chooseBestFeatureToSplit_c(dataSet,
                                                        labelProperty) 
    # 如果无法选出最优分类特征，返回出现次数最多的类别
    if bestFeat == -1:  
        return majorityCnt(classList)


    if labelProperty[bestFeat] == 0:  # 对离散的特征
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (labelsNew[bestFeat])  # 已经选择的特征不再参与分类
        del (labelPropertyNew[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueValue = set(featValues)  # 该特征包含的所有值
        for value in uniqueValue:  # 对每个特征值，递归构建树
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value] = createTree_c(
                splitDataSet_c(dataSet, bestFeat, value), subLabels,
                subLabelProperty)

    # 对连续的特征，不删除该特征，分别构建左子树和右子树
    else:  
        # 设置最优特征的标签
        bestFeatLabel = labels[bestFeat] + '<' + str(bestPartValue)
        # 创建一个以最优分割特征为根节点的子树
        myTree = {bestFeatLabel: {}}
        # 获取标签和特征的类型
        subLabels = labels[:]
        subLabelProperty = labelProperty[:]
        # 构建左子树
        valueLeft = 'Yes'
        myTree[bestFeatLabel][valueLeft] = createTree_c(
            splitDataSet_c(dataSet, bestFeat, bestPartValue, 'L'), subLabels,
            subLabelProperty)
        # 构建右子树
        valueRight = 'No'
        myTree[bestFeatLabel][valueRight] = createTree_c(
            splitDataSet_c(dataSet, bestFeat, bestPartValue, 'R'), subLabels,
            subLabelProperty)
    return myTree

# 进行测试，获得预测结果
def classify_c(inputTree, featLabels, featLabelProperties, testVec):
    # 获得根节点的标签
    firstStr = list(inputTree.keys())[0]
    firstLabel = firstStr
    # 查找是否连续特征
    lessIndex = str(firstStr).find('<')
    # 如果是则获取特征标签
    if lessIndex > -1:
        firstLabel = str(firstStr)[:lessIndex]

    # 获得第一个结点对应树
    secondDict = inputTree[firstStr]
    # 获取特征对应的索引位置
    featIndex = featLabels.index(firstLabel)  # 跟节点对应的特征
    classLabel = None
    # 遍历子树的标签名
    for key in secondDict.keys():
        if featLabelProperties[featIndex] == 0:  # 离散的特征
            if testVec[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict[key], featLabels,
                                           featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict[key]
        
        # 如果是连续的值
        else:
            # 获得标签的分割值
            partValue = float(str(firstStr)[lessIndex + 1:])
            # 如果当前的特征值小于分割值，就进入左子树
            if testVec[featIndex] < partValue:
                # 如果结点是一个子树，就继续递归
                if type(secondDict['Yes']).__name__ == 'dict': 
                    classLabel = classify_c(secondDict['Yes'], featLabels,
                                           featLabelProperties, testVec)
                # 如果是叶子结点，返回类别
                else:
                    classLabel = secondDict['Yes']
            # 进入右子树
            else:
                # 如果结点是一个子树，就继续递归
                if type(secondDict['No']).__name__ == 'dict':
                    classLabel = classify_c(secondDict['No'], featLabels,
                                           featLabelProperties, testVec)
                # 如果是叶子结点，返回类别
                else:
                    classLabel = secondDict['No']
    return classLabel

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == "__main__":
    data, label = getDataSet()

    # range中的数字需要改成特征的个数
    labelProperties = [1 for i in range(30)]  # 属性的类型，0表示离散，1表示连续

    # 交叉验证，这里调用sklearn的接口，几倍交叉验证就填几,保证n/k>3d，最好约等于log(n)，即log(569)
    K = 5
    kf = KFold(n_splits=K)
    m = 0
    score_sum = 0
    # kf会将输入的数据进行分割，并获取训练集合测试集的索引list
    for train_index, test_index in kf.split(data):
        # 训练
        data_litle = data[train_index]
        myTree = createTree_c(list(data_litle), list(label), labelProperties)
        # print(myTree)
        # 测试
        # k是测试的索引
        k = 0
        true = 0
        # 获取数据集
        data_test = load_breast_cancer()
        # 获取数据集中的数据
        data_test_data = data_test.data[test_index]
        # 获取数据集中数据的标签
        targetTemp = data_test.target
        data_test_target = targetTemp[test_index]

        # 遍历
        for x in data_test_data:
            pre = classify_c(myTree, list(label), labelProperties, x)
            real = data_test_target[k]
            k += 1
            if(pre == real):
                true += 1
        # 以准确率为评分
        score = float(true/k)
        score_sum += score
        print("No." + str(m) + " score: " + str(score))
        print("save No." + str(m) + " tree...")
        # 保存树到当前文件夹
        storeTree(myTree, 'tree' + str(m)+ '.txt')
        m += 1

    score_mean = float(score_sum/K)
    print("score_mean: " + str(score_mean))


