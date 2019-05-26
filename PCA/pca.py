from numpy import *


def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 9999999):
    '''
    pca算法
    :param dataMat: 进行pca操作的数据集
    :param topNfeat: 可选参数，即应用的N个特征
    :return:
    '''
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals        #计算，减去原始数据集的平均值
    covMat = cov(meanRemoved, rowvar=0)     #计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))     #计算特征值，特征向量
    eigValInd = argsort(eigVals)        #对特征值进行从小到大的排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]        #根据特征值排序结果的逆序可以得到topNfeat个最大的特征向量
    redEigVects = eigVects[:,eigValInd]         #构成对数据进行转换的矩阵
    lowDDataMat = meanRemoved * redEigVects         #将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]      #计算特征的数目
    for i in range(numFeat):        #在所有特征上进行循环
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])       #对每个特征计算出非NaN的平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal      #替换NaN的值为平均值
    return datMat
import matplotlib.pyplot as plt
if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    # print(shape(lowDMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    # plt.show()
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print(eigVals)
