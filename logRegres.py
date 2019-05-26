'''
梯度上升算法测试函数
求函数f(x) = -x^2 + 4x
'''
def Gradient_Ascent_test():
    def f_prime(x_old):     #f(x)的导数
        return -2 * x_old + 4
    x_old = -1      #初始值，给一个小于x_new的值
    x_new = 0       #梯度上升算法初始值，即从（0，0）开始
    alpha = 0.01        #步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001      #精度，也就是更新阀值
    while abs(x_new - x_old) > presision:       #绝对值
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)      #上面提到的公式
    print(x_new)        #打印最终求解的极值近似值

# if __name__ == '__main__':
#     Gradient_Ascent_test()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import random
from sklearn.linear_model import LogisticRegression
'''
加载数据
'''
def loadDataSet():
    dataMat = []        #创建数据列表
    labelMat = []       #创建标签列表
    fr = open('/Users/user/Desktop/Machine-Learning-master/Logistic/testSet.txt')       #打开文件
    for line in fr.readlines():     #逐行读取
        lineArr = line.strip().split()      #去头尾以及中间的回车、空白字符放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])     #添加数据
        labelMat.append(int(lineArr[2]))        #添加标签
    fr.close()      #关闭文件
    return dataMat, labelMat        #返回数据列表，标签列表

'''
绘制数据集
'''
def ploDataSet():
    dataMat, labelMat = loadDataSet()       #加载数据集
    dataArr = np.array(dataMat)     #转换成numpy的array数组
    n = np.shape(dataMat)[0]        #数据个数，返回dataMat的维数（100，3）元组
    xcord1 = []; ycord1 = []        #正样本
    xcord2 = []; ycord2 = []        #负样本
    for i in range(n):      #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])        #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])        #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)       #添加subplot add_subplot(349)参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块
    ##画图吧，s表示点点的大小，c就是color嘛，marker就是点点的形状哦o,x,*><^,都可以啦
    #alpha,点点的亮度，label，标签啦
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)     #绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha =.5)
    plt.title('DataSet')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

# if __name__ == '__main__':
#     ploDataSet()

'''
sigimoid函数
'''
def sigimoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
梯度上升算法
'''
def gradAscent(dataMatIn, classLabels):
    #数据集， 数据标签
    dataMatrix = np.mat(dataMatIn)      #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()      #转换成numpy的mat,并进行转置
    m,n = np.shape(dataMatrix)      #返回dataMatrix的大小， m为行数，n为列数
    alpha = 0.01       #移动步长，也就是学习速率，控制更新的幅度
    maxCycles = 500     #最大迭代次数
    weights = np.ones((n,1))
    # weights_array = np.array([])
    for k in range(maxCycles):
        h = sigimoid(dataMatrix * weights)      #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        # weights_array = np.append(weights_array, weights)
    # weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA()    #将矩阵转换成数组，返回权重数组 求出回归系数[w0,w1,w2]

# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     print(gradAscent(dataMat, labelMat))

'''
绘制数据集
'''
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker='s',alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)       #返回等差的array
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.show()

# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     weights = gradAscent(dataMat, labelMat)
#     plotBestFit(weights)

'''
改进的随机梯度上升算法
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    #dataMatrix-数据数组；classLabels-数据标签；numIter-迭代次数
    m,n = np.shape(dataMatrix)      #返回dataMatrix的大小，m为行数，n为列数
    weights = np.ones(n)        #参数初始化
    # weights_array = np.array([])

    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            #降低alpha的大小，每次减少1/（j+i），j是迭代次数，i是样本点下标
            randIndex = int(random.uniform(0,len(dataIndex)))       #随机选取样本
            h = sigimoid(sum(dataMatrix[randIndex]*weights))
            #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h      #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归函数
            # weights_array = np.append(weights_array, weights,axis=0)
            del(dataIndex[randIndex])       #删除已经使用的样本
    # weights_array = weights_array.reshape(numIter*m, n)     #改变维度
    return weights    #返回回归系数数组（最优参数）

# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     weights = stocGradAscent1(np.array(dataMat), labelMat)
#     plotBestFit(weights)

'''
绘制回归系数与迭代次数的关系
'''
def plotWeights(weights_array1, weights_array2):
    #weights_array1-回归系数数组1；weights_array2-回归系数数组2
    font = FontProperties(fname = r"/Library/Fonts/Songti.ttc", size = 14)
    #设置汉字格式

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #将fig画布分割成1行1列，不共享x轴和y轴，fig画布的大小为（13，8）
    #当nrows=3，ncols=2时，代表画布被分为六个区域，axs[0][0]表示第一行第一列

    axs[0][0].plot(x1, weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylable_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, color='black')
    plt.setp(axs0_ylable_text, size=20, color='black')
    #绘制w0与迭代次数的关系

    axs[1][0].plot(x1, weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, color='black')
    #绘制w1与迭代次数的关系

    axs[2][0].plot(x1, weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, color='black')
    plt.setp(axs2_ylabel_text, size=20, color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, color='black')
    plt.setp(axs0_ylabel_text, size=20, color='black')
    # 绘制w0与迭代次数的关系

    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, color='black')
    # 绘制w1与迭代次数的关系

    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, color='black')
    plt.setp(axs2_ylabel_text, size=20, color='black')

    # plt.show()

# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet()
#     weights1, weights_array2 = stocGradAscent1(np.array(dataMat), labelMat)
#
#     weights2, weights_array1 = gradAscent(dataMat, labelMat)
#     plotWeights(weights_array1, weights_array2)

'''
使用Python写的Logiditic分类器做预测
'''
def colicTest():
    frTrain = open('/Users/user/Desktop/Machine-Learning-master/Logistic/horseColicTraining.txt')       #打开训练集
    frTest = open('/Users/user/Desktop/Machine-Learning-master/Logistic/horseColicTest.txt')        #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)      #使用随机的上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0])) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100        #错误率计算
    print("测试集错误率为：%.2f%%" % errorRate)

'''
分类函数
'''
def classifyVector(inX, weights):
    prob = sigimoid(sum(inX*weights))
    if prob > 0.5:return  1.0
    else: return 0.0

# if __name__ == '__main__':
#     colicTest()

'''
使用sklearn构建的Logistic回归分类器
'''
def colicSklearn():
    frTrain = open('/Users/user/Desktop/Machine-Learning-master/Logistic/horseColicTraining.txt')  # 打开训练集
    frTest = open('/Users/user/Desktop/Machine-Learning-master/Logistic/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    #solver:优化算法选择参数
    # max_iter:算法收敛最大迭代次数，int类型，仅在正则优化算法为newton-cg, sag和lbfgs才有用，算法收敛的最大迭代次数
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accury = classifier.score(testSet, testLabels) * 100
    print('正确率：%f%%' % test_accury)

if __name__ == '__main__':
    colicSklearn()