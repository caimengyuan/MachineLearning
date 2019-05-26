import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    加载数据
    :param fileName: 文件名
    :return:
        dataMat - 数据矩阵
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')      #字符串类型
        # print(curLine)
        fltLine = list(map(float, curLine))     #转化为float类型
        # print(fltLine)
        dataMat.append(fltLine)
    return dataMat

def plotDataSet(fileName):
    '''
    绘制数据集
    :param fileName: 文件名
    :return: 无
    '''
    dataMat = loadDataSet(fileName)
    n = len(dataMat)        #数据个数
    xcord = []; ycord = []      #样本点
    for i in range(n):
        xcord.append(dataMat[i][1]); ycord.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

# if __name__ == '__main__':
#     filename = 'ex0.txt'
#     plotDataSet(filename)


def bindSplitDataSet(dataSet, feature, value):
    '''
    根据某个特征切分数据集合
    :param dataSet: 数据集合
    :param feature: 带切分的特征
    :param value: 该特征的值
    :return:
        mat0 - 根据第二列切分的数据集合0,大于指定特征值的矩阵
        mat1 - 切分的数据集合1,小于指定特征值的矩阵
    '''
    # print(dataSet[:,feature])
    # print(dataSet[:,feature] <= value)
    # print(np.nonzero(dataSet[:,feature]<=value))
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# if __name__ == '__main__':
#     testMat = np.mat(np.eye(4))
#     mat0, mat1 = bindSplitDataSet(testMat, 1, 0.5)
#     print('原始集合：\n', testMat)
#     print('mat0:\n', mat0)
#     print('mat1:\n', mat1)

def regLeaf(dataSet):
    '''
    生成叶结点
    :param dataSet: 数据集合
    :return:
        目标变量的均值
    '''
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    '''
    误差估计函数
    :param dataSet: 数据集合
    :return:
        目标变量的总方差
    '''
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    '''
    找到数据的最佳二元切分方式函数
    :param dataSet: 数据集合
    :param leafType: 生成叶结点
    :param errType: 误差估计函数
    :param ops: 用户定义的参数构成的元组
    :return:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
    '''
    import types
    #tols允许的误差下降值，tolN切分的最少样本数
    tolS = ops[0]; tolN = ops[1]
    #如果当前所有值相等，则退出。（根据set的特性）
    # s = dataSet[:,-1].T.tolist()[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    #统计数据集合的行m和列n
    m,n = np.shape(dataSet)
    #默认最后一个特征为最佳切分特征，计算其误差估计,方差乘以n
    S = errType(dataSet)
    #分别为最佳误差，最佳特征切分的索引值，最佳特征值
    bestS = float('inf'); bestIndex = 0; bestValue = 0
    #遍历所有特征列
    for featIndex in range(n - 1):
        #遍历所有特征值
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            #根据特征和特征值切分数据集
            mat0, mat1 = bindSplitDataSet(dataSet, featIndex, splitVal)
            #如果数据少于tolN，则退出
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            #计算误差估计
            newS = errType(mat0) + errType(mat1)
            #如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = bindSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #返回最佳切分特征和特征值
    return bestIndex, bestValue

# if __name__ == '__main__':
#     myDat = loadDataSet('ex00.txt')
#     myMat = np.mat(myDat)
#     feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1,4))
#     print(feat)
#     print(val)

def createTree(dataSet, leafType = regLeaf, errType =regErr, ops = (1,4)):
    '''
    树构建函数
    :param dataSet: 数据集合
    :param leafType: 建立叶结点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所有其他参数的元组
    :return:
        retTree - 构建的回归树
    '''
    #选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    #如果没有特征，则返回特征值
    if feat == None: return val
    #回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #分成左数据集和右数据集
    lSet, rSet = bindSplitDataSet(dataSet, feat, val)
    #创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# if __name__ == '__main__':
#     myDat = loadDataSet('ex00.txt')
#     myMat = np.mat(myDat)
#     print(createTree(myMat))

def isTree(obj):
    '''
    判断测试输入变量是否是一棵树
    :param obj: 测试对象
    :return:
        是否是一棵树
    '''
    import types
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    对树进行塌处理（即返回树平均值）
    :param tree: 树
    :return:
        树的平均值
    '''
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    '''
    后剪枝
    :param tree: 树
    :param testData: 测试集
    :return:
        树的平均值
    '''
    #如果测试集为空，则对树进行塌陷处理
    if np.shape(testData)[0] == 0: return getMean(tree)
    #如果有左子树或者右子树，则切分数据集
    if(isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = bindSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #处理左子树（剪枝）
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    #处理右子树（剪枝）
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = bindSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:,-1] - tree['right'],2))
        #计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        #计算合并的误差
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))
        #如果合并的误差小于没有合并的误差，则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else: return tree
    else: return tree

# if __name__ == '__main__':
#     train_filename = 'ex2.txt'
#     train_Data = loadDataSet(train_filename)
#     train_Mat = np.mat(train_Data)
#     tree = createTree(train_Mat)
#     print(tree)
#     test_filename = 'ex2test.txt'
#     test_Data = loadDataSet(test_filename)
#     test_Mat = np.mat(test_Data)
#     print(prune(tree, test_Mat))

'''
模型树,构造线性回归函数
'''
def linearSolve(dataSet):
    '''
    将数据集格式化成目标变量Y和自变量X，x，y用于执行简单的线性回归
    :param dataSet:
    :return: 回归系数
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
    X[:, 1:n] = dataSet[:, 0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value od ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    '''
    类似regLeaf()，当数据不再需要切分的时候负责生成叶结点的模型
    :param dataSet:
    :return: 回归系数
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''
    在给定的数据集上计算误差，类似regError()
    :param dataSet:
    :return: 返回Y和yHat之间的平方误差
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power((Y - yHat), 2))

# if __name__ == '__main__':
#     myMat2 = np.mat(loadDataSet('exp2.txt'))
#     print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))


'''
用树回归进行预测
'''
def regTreeEval(model, inDat):
    '''
    对回归树叶结点预测
    :param model:
    :param inDat:
    :return:
    '''
    return float(model)

def modelTreeEval(model, inDat):
    '''
    对模型树结点预测
    :param model:
    :param inDat:
    :return:
    '''
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat