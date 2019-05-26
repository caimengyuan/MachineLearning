from numpy import *
import operator
from os import listdir
from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k): # 分类的输入向量，训练样本集，标签向量，最近邻的数目
    dataSetSize = dataSet.shape[0] #  0是行 1是列 shape[0]读取dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # title 重复inx数组

    # 输入向量到样本集的距离
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #返回数组值从小到大的索引值

    classCount={}
    # 返回前k个训练样本集
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True) #itemgetter获取对象的第一个值
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) #获取文本的行数
    returnMat = zeros((numberOfLines,3)) #创建二维矩阵存储训练样本数据，一共有nunberOfLines行，3列
    classLabelVector = [] #创建一维数组存放训练样本标签
    index = 0

    for line in arrayOLines:
        line = line.strip() #去掉回车
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        # labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# datingDataMat, datingLabels = file2matrix('/Users/user/Desktop/datingTestSet.txt')
# 归一化数值
def autoNorm(dataSet):
    minVals = dataSet.min(0) # minVals是每一列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) # 将minVals复制成m行1列的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

# 测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('/Users/user/Desktop/Data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  # 选取百分之十的样本来测试
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with : %d,the real answer is : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is :%f'% (errorCount/float(numTestVecs)))

# 约会网站的预测函数
def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('/Users/user/Desktop/Data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print('you will probably like this person: ', resultList[classifierResult - 1])

# 将图像转化为向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # with open(filename) as fr:
    for a in range(32):
        lineStr = fr.readline()
        for b in range(32):
            returnVect[0, 32*a+b] = int(lineStr[b])
    return returnVect

#
# testVector = img2vector('/Users/user/Desktop/Data/digits/testDigits/0_19.txt')
# print(testVector[0,0:31])

def handwritingClassTest():
    hwLabels= []
    trainingFileList = listdir('/Users/user/Desktop/Data/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('/Users/user/Desktop/Data/digits/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('/Users/user/Desktop/Data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/Users/user/Desktop/Data/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classfierResult came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
        if(classNumStr != classifierResult):
            errorCount += 1.0
    print('\n the total number of errors is: %d' % errorCount)
    print('\n the total error rate is: %f' % (errorCount/float(mTest)))

# print(handwritingClassTest())

