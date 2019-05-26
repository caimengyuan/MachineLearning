import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def loadDataSet(fileName):
    '''
    加载数据
    :param fileName: 文件名
    :return:
        xArr - x数据集
        yArr - y数据集
    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = [];
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr

def plotDataSet():
    '''
    绘制数据集
    :return: 无
    '''
    xArr, yArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/ex0.txt')
    n = len(xArr)       #数据个数
    xcord = []; ycord = []      #样本点
    for i in range(n):
        xcord.append(xArr[i][1]); ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s = 20, c = 'blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

# if __name__ == '__main__':
#     plotDataSet()

def standRegres(xArr, yArr):
    '''
    计算回归系数w
    :param xArr: x数据集
    :param yArr: y数据集
    :return:
        ws：回归系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat     #根据推导公式计算回归系数
    if np.linalg.det(xTx) == 0.0:       #矩阵求行列式
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def plotRegression():
    '''
    绘制回归曲线和数据点
    :return: 无
    '''
    xArr, yArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws       #计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1], yHat, c='red')
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

# if __name__ == '__main__':
#     plotRegression()

'''
比较预测值和真实值的相关性
'''
# if __name__ == '__main__':
#     xArr, yArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/ex0.txt')
#     ws = standRegres(xArr,yArr)
#     xMat = np.mat(xArr)
#     yMat = np.mat(yArr)
#     yHat = xMat * ws
#     print(np.corrcoef(yHat.T, yMat))    #向量的相似程度

def lwlr(testPoint, xArr, yArr, k = 1.0):
    '''
    使用局部加权线性回归计算回归系数w
    :param testPoint: 测试样本点
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k，自定义参数
    :return:
        ws-回归系数
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))   #创建权重对角矩阵
    for j in range(m):      #遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0 :
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    '''
    局部加权线性回归测试
    :param testArr: 测试数据集
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k，自定义参数
    :return:
        ws - 回归系数
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):  #对每个样本点进行预测
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotlwlrRegression():
    '''
    绘制多条局部加权回归曲线
    :return: 无
    '''
    font = FontProperties(fname = r"/Library/Fonts/Songti.ttc", size = 14)
    xArr, yArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/ex0.txt')
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    srtInd = xMat[:,1].argsort(0)   #排序返回索引值
    xSort = xMat[srtInd][:,0,:]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10,8))
    axs[0].plot(xSort[:,1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:,1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:,1], yHat_3[srtInd], c='red')
    axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

# if __name__ == '__main__':
#     plotlwlrRegression()

def rssError(yArr, yHatArr):
    '''
    误差大小评价函数
    :param yArr: 真实数据
    :param yHatArr: 预测数据
    :return:
        误差大小
    '''
    return ((yArr - yHatArr) ** 2).sum()

# if __name__ == '__main__':
#     abX, abY = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/abalone.txt')
#     print("训练集与测试集相同：局部加权线性回归，核k的大小对预测的影响：")
#     yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
#     yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
#     yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
#     print('k=0.1时，误差大小为：', rssError(abY[0:99], yHat01.T))
#     print('k=1时，误差大小为：', rssError(abY[0:99], yHat1.T))
#     print('k=10时，误差大小为：', rssError(abY[0:99], yHat10.T))
#     print('')
#     print('训练集与测试集不同：局部加权线性回归，核k的大小是越小越好吗？更换数据集，测试结果如下：')
#     yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
#     yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
#     yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
#     print('k=0.1时，误差大小为：', rssError(abY[100:199], yHat01.T))
#     print('k=1时，误差大小为：', rssError(abY[100:199], yHat1.T))
#     print('k=10时，误差大小为：', rssError(abY[100:199], yHat10.T))
#     print('')
#     print('训练集与测试集不同：简单的线性回归与k=1时的局部加权线性回归对比：')
#     print('k=1时，误差大小为：',rssError(abY[100:199],yHat1.T))
#     ws = standRegres(abX[0:99], abY[0:99])
#     yHat = np.mat(abX[100:199]) * ws
#     print('简单的线性回归误差大小：', rssError(abY[100:199], yHat.T.A))

def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    岭回归
    :param xMat: x数据集
    :param yMat: y数据集
    :param lam: 缩减系数
    :return:
        ws-回归系数
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('矩阵为奇异矩阵，不能转置')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''
    岭回归测试
    :param xArr: x数据集
    :param yArr: y数据集
    :return:
        wMat - 回归系数矩阵
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat, axis = 0)     #行与行操作，求均值
    yMat = yMat - yMean     #数据减去均值
    xMeans = np.mean(xMat, axis = 0)    #行与行操作，求均值
    xVar = np.var(xMat, axis=0)     #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar       #数据减去均值除以方差实现标准化
    numTestPts = 30     #30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))        #初始回归系数矩阵
    for i in range(numTestPts):     #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))    #lambda以e的指数变化，最初是一个非常小的数
        wMat[i,:] = ws.T        #计算回归系数矩阵
    return wMat

def plotwMat():
    '''
    绘制岭回归系数矩阵
    :return:
    '''
    font = FontProperties(fname = r"/Library/Fonts/Songti.ttc", size = 14)
    abX, abY = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)',FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()

# if __name__ == '__main__':
#     plotwMat()

def regularize(xMat, yMat):
    '''
    数据标准化
    :param xMat: x数据集
    :param yMat: y数据集
    :return:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
    '''
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)
    inyMat = yMat - yMean
    inMeans = np.mean(inxMat, 0)
    inVar = np.var(inxMat, 0)
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    '''
    前向逐步线性回归
    :param xArr: x输入值
    :param yArr: y预测数据
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
        returnMat - numIt次迭代的回归系数矩阵
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))    #初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))   #初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):      #迭代numIt次
        # print(ws.T)     #打印当前回归系数矩阵
        lowestError = float('inf');     #正无穷
        for j in range(n):      #遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign     #微调回归系数
                yTest = xMat * wsTest       #计算预测值
                rssE = rssError(yMat.A, yTest.A)    #计算平方误差
                if rssE < lowestError:      #如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T       #记录numIt次迭代的回归系数矩阵
    return returnMat

def plotstageWiseMat():
    '''
    绘制岭回归系数矩阵
    :return:
    '''
    font = FontProperties(fname = r"/Library/Fonts/Songti.ttc", size = 14)
    xArr, yArr = loadDataSet('/Users/user/Desktop/Machine-Learning-master/Regression/abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()

# if __name__ == '__main__':
#     plotstageWiseMat()





import numpy as np
from bs4 import BeautifulSoup
import random

def scrapePage(retX, retY, inFile, yr, numPce, origprc):
    '''
    从页面读取数据，生成retX和retY列表
    :param retX: 数据X
    :param retY: 数据y
    :param inFile: HTML文件
    :param yr: 年份
    :param numPce: 乐高部件数目
    :param origprc: 原价
    :return: 出品年份 部件数目 是否为全新 原价 售价（二手交易）
    '''
    #打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    #根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all("table", r = "%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        #查找是否有全新标签
        if(lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        #查找是否已经标志出售，我们只手机已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            #解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping','')
            sellingPrice = float(priceStr)
            #去掉不完整的套装价格
            if sellingPrice > origprc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origprc, sellingPrice))
                retX.append([yr, numPce, newFlag, origprc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = '%d' % i)

def setDataCollect(retX, retY):
    '''
    依次读取六种乐高套装的数据，并生成数据矩阵
    :param retX:
    :param retY:
    :return:
    '''
    #2006年的乐高8288，部件数目800，原价49.99
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego10179.html', 2006, 5195, 499.99)
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego10181.html', 2006, 3428, 199.99)
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego10189.html', 2006, 5922, 299.99)
    scrapePage(retX, retY, '/Users/user/Desktop/Machine-Learning-master/Regression/lego/lego10196.html', 2006, 3263, 249.99)

# if __name__ == '__main__':
#     lgX = []
#     lgY = []
#     setDataCollect(lgX, lgY)

def useStandRegres():
    '''
    使用简单的线性回归
    :return: 无
    '''
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    data_num, features_num = np.shape(lgX)
    lgX1 = np.mat(np.ones((data_num, features_num + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0],ws[1],ws[2],ws[3],ws[4]))

# if __name__ == '__main__':
#     useStandRegres()

def crossValidation(xArr, yArr, numVal = 10):
    '''
    交叉验证岭回归
    :param xArr: x数据集
    :param yArr: y数据集
    :param numVal: 交叉验证次数
    :return:
        wMat - 回归系数矩阵
    '''
    m = len(yArr)       #统计样本个数
    indexList = list(range(m))      #生成索引值列表
    errorMat = np.zeros((numVal, 30))       #create error mat 30columns numVal rows
    for i in range(numVal):     #交叉验证numVal次
        trainX = []; trainY = []        #训练集
        testX = []; testY = []      #测试集
        random.shuffle(indexList)       #打乱次序
        for j in range(m):      #划分数据集：90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)    #获得30个不同lambda下的岭回归系数
        for k in range(30):     #遍历所有岭回归系数
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)    #测试集
            meanTrain = np.mean(matTrainX,0)    #测试集均值
            varTrain = np.var(matTrainX,0)      #测试集方差
            matTestX = (matTestX - meanTrain) / varTrain        #测试集标准化
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)    #根据ws预测y值
            errorMat[i,k] = rssError(yEst.T.A, np.array(testY))     #统计误差
    meanErrors = np.mean(errorMat,0)        #计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))        #找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]       #找到最佳回归系数
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX      #数据经过标准化，因此需要还原
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))

# if __name__ == '__main__':
#     lgX = []
#     lgY = []
#     setDataCollect(lgX, lgY)
#     crossValidation(lgX, lgY)
