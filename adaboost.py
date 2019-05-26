import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadSimpData():
    '''
    创建单层决策树的数据集
    :return:
        dataMat - 数据矩阵
        classLabels - 数据标签
    '''
    datMat = np.matrix([
        [1. , 2.1],
        [1.5 , 1.6],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]]
    )
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def showDataSet(dataMat, labelMat):
    '''
    数据可视化
    :param dataMat:
    :param labelMat:
    :return: 无
    '''
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

# if __name__ == '__main__':
#     dataArr, classLables = loadSimpData()
#     showDataSet(dataArr, classLables)

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    单层决策树分类函数
    :param dataMatrix: 数据矩阵
    :param dimen: 第dimen列，也就是第几个特征
    :param threshVal: 阀值
    :param threshIneq: 标志
    :return:
        retArray - 分类结果
    '''
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0   #如果小于阀值，则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStrump(dataArr, classLabels, D):
    '''
    找到数据集上最佳的单层决策树
    :param dataArr:
    :param classLabels:
    :param D: 样本权重
    :return:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    '''
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')  #最小误差初始化为正无穷大
    for i in range(n):  #遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()  #找到特征值中最大和最小
        stepSize = (rangeMax - rangeMin) / numSteps     #计算步长
        for j in range(-1, int(numSteps) + 1):
            for ineuqal in ['lt', 'gt']:    #大于和小于的情况，均遍历 lt：less than, gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)    #计算阀值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, ineuqal)    #计算分类结果
                errArr = np.mat(np.ones((m,1)))     #初始化误差矩阵
                errArr[predictedVals == labelMat] = 0       #分类正确的，误差为0
                weightedError = D.T * errArr        #计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, ineuqal, weightedError))
                if weightedError < minError:        #找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal     #阀值
                    bestStump['ineq'] = ineuqal     #标志
    return bestStump, minError, bestClasEst

# if __name__ == '__main__':
#     dataArr, classLabels = loadSimpData()
#     D = np.mat(np.ones((5,1)) / 5)
#     bestStump, minError, bestClaEst = buildStrump(dataArr, classLabels, D)
#     print('bestStump:\n', bestStump)
#     print('minEroor:\n', minError)
#     print('bestClasEst:\n', bestClaEst)

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    #初始化权重
    D = np.mat(np.ones((m,1)) / m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        #构建单层决策树
        bestStump, error, classEst = buildStrump(dataArr, classLabels, D)
        # print("D:", D.T)
        #计算弱学习算法权重alpha，使error不等于0，因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha      #存储弱学习算法权重
        weakClassArr.append(bestStump)      #存储单层决策树
        # print("classEst: ", classEst.T)
        #计算e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()     #根据样本权重公式，更新样本权重
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst     #记录每个数据点的类别估计值
        # print("aggClassEst:", aggClassEst.T)
        #计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        # print("total error:", errorRate)
        if errorRate == 0.0: break      #误差为0，退出循环
    return weakClassArr, aggClassEst

# if __name__ == '__main__':
#     dataArr, classLabels = loadSimpData()
#     weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
#     print(weakClassArr)
#     print(aggClassEst)

def adaClassify(datToClass, classifierArr):
    '''
    AdaBoost分类函数
    :param datToClass: 待分类样例
    :param classifierArr: 训练好的分类器
    :return: 分类结果
    '''
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print(adaClassify(([0,0],[5,5]), weakClassArr))


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# if __name__ == '__main__':
#     dataArr, labelArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/AdaBoost/horseColicTraining2.txt')
#     weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr)
#     testArr, testLabelArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/AdaBoost/horseColicTraining2.txt')
#     print(weakClassArr)
#     predictions = adaClassify(dataArr, weakClassArr)
#     errArr = np.mat(np.ones((len(dataArr), 1)))
#     print('训练集的错误率：%.3f%%' % float(errArr[predictions != np.mat(labelArr).T].sum() / len(dataArr) * 100))
#     predictions = adaClassify(testArr, weakClassArr)
#     errArr = np.mat(np.ones((len(testArr), 1)))
#     print('测试集的错误率：%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))


def plotROC(predStrengths, classLabels):
    '''
    绘制ROC
    :param predStrengths: 分类器的预测强度
    :param classLabels: 类别
    :return: 无
    '''
    font = FontProperties(fname = r"/Library/Fonts/Songti.ttc", size = 14)
    cur = (1.0, 1.0)        #绘制光标的位置
    ySum = 0.0      #用于计算AUC:曲线下的面积，分类器的平均性能
    numPosClas = np.sum(np.array(classLabels) == 1.0)   #统计正类的数量
    yStep = 1 / float(numPosClas)   #y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)    #x轴步长

    sortedIndicies = predStrengths.argsort()    #预测强度排序,返回数组值从小到大的索引值

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]      #高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')    #绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)    #更新绘制光标的位置
    ax.plot([0,1], [0,1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties = font)
    plt.xlabel('假阳率', FontProperties = font)
    plt.ylabel('真阳率', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为：', ySum * xStep)
    plt.show()

# if __name__ == '__main__':
#     dataArr, labelArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/AdaBoost/horseColicTraining2.txt')
#     weakClassAr, aggClassEst = adaBoostTrainDS(dataArr, labelArr)
#     plotROC(aggClassEst.T, labelArr)