import matplotlib.pyplot as plt
import numpy as np
import random

class optStruct:
    '''
    数据结构，维护所有需要操作的值
    dataMatln-数据矩阵
    classLabels-数据标签
    C-松弛变量
    toler-容错率
    '''
    def __init__(self, dataMatln, classLabels, C, toler):
        self.X = dataMatln      #数据矩阵
        self.labelMat = classLabels     #数据标签
        self.C = C      #松弛变量
        self.tol = toler        #容错率
        self.m = np.shape(dataMatln)[0]     #数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m,1)))      #根据矩阵行数初始化alpha参数为0
        self.b = 0      #初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m,2)))      #根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值

def loadDataSet(fileName):
    '''
    读取数据
    :returns
        dataMat-数据矩阵
        labelMat-数据标签
    '''
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def calcEk(oS, k):
    '''
    计算误差
    :param oS: -数据结构
    :param k: -标号为k的数据
    :return:
        Ek-标号为k的数据误差
    '''
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    '''
    随机选择alpha_j的索引值
    :param i: alpha_i的索引值
    :param m: alpha参数个数
    :return: alpha_j的索引值
    '''
    j = i       #选择一个不等于i的j
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    '''
    内循环启发方式2
    :param i: 标号为i的数据的索引值
    :param oS: 数据结构
    :param Ei: 标号为i的数据误差
    :return:
        j,maxK-标号为j或maxK的数据索引值
        Ej-标号为j的数据误差
    '''
    maxK = -1; maxDeletaE = 0; Ej = 0       #初始化
    oS.eCache[i] = [1,Ei]       #根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]   #返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  #有不为0的误差
        for k in validEcacheList:   #遍历，找到最大的Ek
            if k == i:continue  #不计算i，浪费时间
            Ek = calcEk(oS, k)  #计算Ek
            deltaE = abs(Ei - Ek)   #计算|Ei-Ek|
            if (deltaE > maxDeletaE):  #找到maxDeltaE
                maxK = k; maxDeletaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #没有不为0的误差
        j = selectJrand(i, oS.m)    #随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  #计算Ej
    return j, Ej

def updateEk(oS, k):
    '''
    计算Ek， 并更新误差缓存
    :param oS: 数据结构
    :param k: 标号为k的数据的索引值
    :return: 无
    '''
    Ek = calcEk(oS, k)  #计算Ek
    oS.eCache[k] = [1,Ek]   #更新误差缓存

def clipAlpha(aj, H, L):
    '''
    修剪alpha_i
    :param aj: alpha_i的值
    :param H: alpha上限
    :param L: alpha下限
    :return:
        aj-修剪后的alpha_j的值
    '''
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def innerL(i, oS):
    '''
    优化的SMO算法
    :param i: 标号为i的数据的索引值
    :param oS: 数据结构
    :return:
        1-有任意一对alpha值发送变化
        0-没有任意一对alpha值发生变化或变化太小
    '''
    #步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    #优化alpha，设定一定的容错率
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        #保存更新前的alpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        #步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("l==H")
            return 0
        #步骤3：计算eta
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        #步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H ,L)
        #更新Ej至误差缓存
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.000001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS, i)
        #步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[i,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        #步骤8：根据b_1和b_2更新b
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else:oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLables, C, toler, maxIter):
    '''
    完整的线性SMO算法
    :param dataMatln: 数据矩阵
    :param classLables: 数据标签
    :param C: 松弛变量
    :param toler: 容错率
    :param maxIter: 最大迭代次数
    :return:
        oS.b-SMO算法计算的b
        oS.alphas-SMO算法计算的alphas
    '''
    oS = optStruct(np.mat(dataMatIn), np.mat(classLables).transpose(), C, toler)    #初始化数据结构
    iter = 0    #初始化当前迭代次数
    entireSet = True; alphaPairsChanged = 0
    #遍历整个数据集，alpha没有更新或者超过最大迭代次数，则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #遍历整个数据集
        if entireSet:
            for i in range(oS.m):
                #使用优化的SMO算法
                alphaPairsChanged += innerL(i,oS)
                print("全样本遍历：第%d次迭代样本：%d，alpha优化次数：%d" %(iter, i, alphaPairsChanged))
            iter += 1
        #遍历非边界值
        else:
            #nonzero-获取数组（矩阵）中不为0的下标
            #遍历不在边界0和C的alpha
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历：第%d次迭代 样本：%d，alpha优化次数：%d" % (iter, i, alphaPairsChanged))
            iter += 1
        #遍历一次后改为非边界遍历
        if entireSet:
            entireSet = False
        #如果样本没有更新，计算全样本遍历
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("迭代次数：%d" % iter)
    #返回SMO算法计算的b和alphas
    return oS.b, oS.alphas

def showClassifer(dataMat, classLabels, w, b):
    '''
    分类结果可视化
    :param dataMat: 数据矩阵
    :param classLabels: 数据标签
    :param w: 直线法向量
    :param b: 直线解决
    :return: 无
    '''
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_up = np.array(data_plus)
    data_minus_up = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_up)[0], np.transpose(data_plus_up)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_up)[0], np.transpose(data_minus_up)[1], s=30, alpha=0.7)
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x,y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidths=1.5, edgecolors='red')
    plt.show()

def calcWs(alphas, dataArr, classLabels):
    '''
    计算w
    :param alphas: alphas值
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :return: 计算得到的w
    '''
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w



# if __name__ == '__main__':
#     dataArr, classLabels = loadDataSet('/Users/user/Desktop/Machine-Learning-master/SVM/testSet.txt')
#     b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
#     w = calcWs(alphas, dataArr, classLabels)
#     showClassifer(dataArr, classLabels, w, b)
