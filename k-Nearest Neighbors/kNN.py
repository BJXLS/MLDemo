import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 使用欧式距离计算距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5

    # 获取排序index
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)

    classCount = {}
    # 取前K个
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 计算对应分类
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 降序排序 
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True,)
    return sortedClassCount[0][0]

def classify(thisOne, dataSet, labels, k):
    distances = []
    dataSetSize = dataSet.shape[0]




# 文字处理模块
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    print(numberOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # print(type(line))
        line = line.strip()
        listFromLine = line.split('\t')
        # print(listFromLine)
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def figure(DataMat, Label):
    import matplotlib.pyplot as plt
    # 第3个是大小 第4个是颜色
    plt.scatter(DataMat[:, 1], DataMat[:, 2], 15.0*np.array(Label), 15.0*np.array(Label))
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    DataMat, Labels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(DataMat)
    m = normMat.shape[0]
    
    # 这是为了留下前100个数据进行测试
    hoRatio = 0.10
    numTestVecs = int(m * hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, : ], normMat[numTestVecs: m, : ], Labels[numTestVecs: m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, Labels[i]))
        if (classifierResult != Labels[i]):
            errorCount += 1.0
        print("the total error rate is: %f" % (errorCount/ float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    DataMat, Labels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(DataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, Labels, 3)
    print("you will probably like this person: ", resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    from os import listdir
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits\\%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits\\%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier cam back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

if __name__ == "__main__":
    # DataMat, Label = file2matrix("datingTestSet2.txt")
    # # figure(DataMat, Label)
    # normMat, ranges, minVals = autoNorm(DataMat)
    # print(normMat.shape[1])
    # print(normMat[0, 0])
    # print(ranges)
    # print(minVals)
    # datingClassTest()
    # classifyPerson()  
    # testVector = img2vector('testDigits\\0_13.txt')
    # print(testVector)
    handwritingClassTest()