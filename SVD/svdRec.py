from numpy import *
from numpy.linalg import linalg as la

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def eulidSim(inA, inB):     #计算欧式距离
    return 1.0 / (1.0 + la.norm(inA - inB))     #计算二范式

def pearsSim(inA, inB):         #皮尔逊相关系数
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):       #余弦相似度
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)

# if __name__ == '__main__':
#     Data = loadExData()
#     U, Sigma, VT = la.svd(Data)
#     # print(Sigma)
#     myMat = mat(loadExData())
#     print(eulidSim(myMat[:,0], myMat[:,4]))
#     print(eulidSim(myMat[:,0], myMat[:,0]))
#     print(cosSim(myMat[:,0], myMat[:,4]))
#     print(cosSim(myMat[:,0], myMat[:,0]))
#     print(pearsSim(myMat[:,0], myMat[:,4]))
#     print(pearsSim(myMat[:,0], myMat[:,0]))

def standEst(dataMat, user, simMeas, item):
    '''
        给定相似度计算方法方法的情况下，用户对物品的估计评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 物品编号
    :param item: 相似度计算方法
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas = cosSim, estMethod = standEst):
    '''
        推荐引擎
    :param dataMat:
    :param user:
    :param N:
    :param simMeas:
    :param estMethod:
    :return:
    '''
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]

