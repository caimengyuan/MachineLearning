import numpy as np
import random

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    '''
    计算两个向量的欧式距离，是个距离函数
    :param vecA:
    :param vecB:
    :return:
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, K):
    '''
    为给定数据集构建一个包含k个随机质心的集合
    :param dataSet:
    :param K:
    :return:
        包含k个随机质心的集合
    '''
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((K, n)))
    for j in range(n):      #构建族质心
        minJ = min(dataSet[:,j])        #数据集每一维的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)       #整个数据集的范围
        centroids[:,j] = minJ + rangeJ * np.random.rand(K,1)       #随机点在边界范围以内
    return centroids

# if __name__ == '__main__':
#     dataMat = np.mat(loadDataSet('testSet.txt'))
#     print(min(dataMat[:, 0]))
#     min(dataMat[:,1])
#     max(dataMat[:,1])
#     max(dataMat[:,0])
#     print(randCent(dataMat, 2))

def KMeans(dataSet, K, distMeas=distEclud, createCent=randCent):
    '''
    K-均值算法
    :param dataSet: 数据集
    :param K: 簇的数目
    :param distMeas: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''
    m = np.shape(dataSet)[0]    #数据集中数据点的总数
    clusterAssment = np.mat(np.zeros((m,2)))    #储存每个点的族分配结果，包含两列：一列记录簇的索引值、一列存储误差
    centorids = createCent(dataSet, K)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf'); minIndex = -1
            for j in range(K):      #寻找最近的质心
                distJI = distMeas(centorids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        # print(centorids)
        #更新质心的位置
        for cent in range(K):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centorids[cent,:] = np.mean(ptsInClust, axis=0)
    return centorids, clusterAssment

# if __name__ == '__main__':
#     dataMat = np.mat(loadDataSet('testSet.txt'))
#     myCentroids, clustAssing = KMeans(dataMat,4)
#     print(myCentroids, clustAssing)

def biKmeans(dataSet, K, distMeas=distEclud):
    '''
    二分k均值聚类算法
    :param dataSet:
    :param K:
    :param distMeas: 距离函数
    :return: 聚类结果
    '''
    m = np.shape(dataSet)[0]
    clusterAssmen = np.mat(np.zeros((m,2)))     #存储数据集中每个点的簇分配结果及平方误差
    centroid0 = np.mean(dataSet, axis = 0).tolist()[0]      #整个数据集的质心
    centList = [centroid0]      #列表，保留所有的质心
    for j in range(m):      #遍历数据集中所有的点计算每个点到质心的误差值
        clusterAssmen[j,1] = distMeas(np.mat(centroid0), dataSet[j,:]) ** 2
    while (len(centList) < K):      #对簇进行划分，直到得到想要的簇数目为止
        lowestSSE = float('inf')
        for i in range(len(centList)):      #遍历所有的簇来决定最佳的簇进行划分，通过比较SSE
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssmen[:,0].A == i)[0],:]      #每个簇中的所有点看成一个小的数据集
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2, distMeas)      #对ptsInCurrCluster输入到KMeans中进行处理
            sseSplit = sum(splitClustAss[:,1])      #误差值
            sseNotSplit = sum(clusterAssmen[np.nonzero(clusterAssmen[:,0].A != i)[0],1])
            print("sseSplit, and notSplit", sseSplit, sseNotSplit)
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCenToSplit is :', bestCentToSplit)
        print('the len of bestClusAss is :', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssmen[np.nonzero(clusterAssmen[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return np.mat(centList), clusterAssmen


# if __name__ == '__main__':
#     datMat3 = np.mat(loadDataSet('testSet2.txt'))
#     cenList, myNewAssments = biKmeans(datMat3,3)


import urllib
import urllib.request
import json

def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

if __name__ == '__main__':
    geoResults = geoGrab('1 VA Center', 'Auguest, ME')
