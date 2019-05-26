class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None        #链接相似的元素项
        self.parent = parentNode
        self.children = {}      #存放节点的子节点

    def inc(self, numOccur):        #给count变量增加给定值
        self.count += numOccur

    def disp(self, ind=1):      #将树以文本形式显示
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

# if __name__ == '__main__':
#     rootNode = treeNode('pyramid', 9, None)
#     rootNode.children['eye'] = treeNode('eye', 13, None)
#     rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
#     rootNode.disp()

def createTree(dataSet, minSup = 1):
    '''
    构建FP树
    :param dataSet: 数据集
    :param minSup: 最小支持度
    :return:
    '''
    headerTable = {}        #头指针
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):        #删掉出现次数小于最小支持度的项
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:       #如果没有元素项满足要求，则退出
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]     #对头指针表扩展以保存计数值和指向每种类型第一个元素项的指针
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():      #遍历数据集，只考虑频繁项集
        localD = {}
        for item in tranSet:        #根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orederedItems = [v[0] for v in sorted(localD.items(), key = lambda p:p[1], reverse=True)]
            updateTree(orederedItems, retTree, headerTable, count)      #使用排序后的频率项集对树进行填充
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:     #测试事务中的第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)        #更新该元素项的计数
    else:       #创建一个新的treeNode并将其作为一个子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:       #头指针更新以指向新的节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)       #不断迭代调用自身，每次调用会去掉列表的第一个元素

def updateHeader(nodeToTest, targetNode):
    '''
    确保节点链接指向树中该元素项的每个一个实例
    :param nodeToTest:
    :param targetNode:
    :return:
    '''
    while(nodeToTest.nodeLink != None):     #nodeLink头指针
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    '''
        迭代上溯整棵树
    :param leafNode: 指针
    :param prefixPath: 存储遍历的元素项
    :return:
    '''
    if leafNode.parent != None:     #上溯
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    '''
         条件模式集
    :param basePat: 给定元素项
    :param treeNode: 头指针表中给定元素项的指针
    :return:
        条件模式集字典
    '''
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[0])]       #对头指针元素项按照其出现频率进行排序，从小到大
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)     #将每一个频繁项添加到频繁项集列表
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)      #从条件模式基来构建条件FP树
        if myHead != None:
            print('onditional tree for:', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)      #挖掘条件FP树



if __name__ == '__main__':
    simpDat = loadSimpDat()
    # print(simpDat)
    initSet = createInitSet(simpDat)
    # print(initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    # myFPtree.disp()
    # print(myHeaderTab)
    # print(findPrefixPath('x', myHeaderTab['x'][1]))
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)