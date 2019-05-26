def loadDataSet():
    '''
    用于测试的简单数据集
    :return:
    '''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    构建集合C1，C1是大小为1的所有候选项集的集合
    :param dataSet:
    :return:
    '''
    C1 = []     #存储不重复的项值
    for transaction in dataSet:     #从输入数据中提取候选集的列表
        for item in transaction:
            if [item] not in C1:        #添加只包含该物品项的一个列表
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))       #frozenset数据类型，"冰冻"集合：不可改变的，用户不能修改

def scanD(D, CK, minSupport):
    '''
    从C1生成L1
    :param D: 数据集
    :param CK: 候选项集
    :param minSupport: 感兴趣项集的最小支持度
    :return:
        返回一个包含支持度的字典以备后用
    '''
    ssCnt = {}      #创建空字典
    for tid in D:   #遍历数据集
        for can in CK:      #遍历候选集
            if can.issubset(tid):       #集合是记录的一部分
                # if not ssCnt.has_key(can): ssCnt[can] = 1       #增加字典中对应的计数值
                if can not in ssCnt:
                    ssCnt[can] = 1
                else: ssCnt[can] += 1       #字典的建是集合
    numItems = float(len(D))
    retList = []    #最小支持度要求的集合
    supportData = {}        #最频繁项集的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems       #支持度
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

# if __name__ == '__main__':
#     dataSet = loadDataSet()
#     C1 = createC1(dataSet)
#     print(C1)
#     # D = map(set, dataSet)
#     # print(D)
#     L1, suppData0 = scanD(dataSet, C1, 0.5)
#     print(L1)

def aprioriGen(LK, K):
    '''
    创建候选集
    :param LK: 频繁项集列表
    :param K: 项集元素
    :return:
    '''
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(LK[i])[:K-2]; L2 = list(LK[j])[:K-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    '''
    生成候选项集的列表
    :param dataSet: 数据集
    :param minSupport: 支持度
    :return:
    '''
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]      #L中会包含L1,L2,L3...
    K = 2
    while (len(L[K-2]) > 0):
        CK = aprioriGen(L[K-2], K)      #创建C2候选项集
        LK, supK = scanD(D, CK, minSupport)     #由L1得到L2 或者L2得到L3等等
        supportData.update(supK)
        L.append(LK)        #频繁项集
        K += 1
    return L, supportData

# if __name__ == '__main__':
#     dataSet = loadDataSet()
#     L, suppData = apriori(dataSet)
#     print(L)
#     print(aprioriGen(L[0], 2))

def generateRules(L, supportData, minConf=0.7):
    '''
        主函数，调用rulesFromConseq和calcConf函数
    :param L: 频繁项集列表
    :param supportData: 包含频繁项集支持数据的字典
    :param minConf: 最小可信度阀值
    :return:
        包含可信度的规则列表，后面可以基于可信度对其进行排序
    '''
    bigRuleList = []
    for i in range(1, len(L)):      #遍历L中的每个频繁项集
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]        #对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):     #项集元素多于2，对它进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:       #计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
        对规则进行评估，找到满足最小可信度要求的规则
    :param freqSet: 前件
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return:
        返回满足最小可信度要求的规则列表
    '''
    prunedH = []        #保存规则
    for conseq in H:        #遍历H中所有项集并计算可信度值
        conf = supportData[freqSet] / supportData[freqSet - conseq]     #导入支持度数据，节省大量计算时间
        if conf >= minConf:     #满足最小可信度
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)     #输出到屏幕
            brl.append((freqSet-conseq, conseq, conf))      #更新可信度规则列表
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
        生成候选规则集合，从最初的项集中生成更多的关联规则
    :param freqSet: 频繁项集
    :param H: 出现在规则右部的元素列表
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    '''
    m = len(H[0])
    if (len(freqSet) > (m + 1)):        #频繁项集是否大到可以移除大小为m的子集
        Hmp1 = aprioriGen(H, m + 1)     #生成无重复组合，是下一次迭代的H列表
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)       #测试可信度确定规则是否满足要求
        if (len(Hmp1) > 1):         #不止一条规则满足要求
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)       #判断是否可以进一步组合这些规则

# if __name__ == '__main__':
#     dataSet = loadDataSet()
#     L, suppData = apriori(dataSet, minSupport=0.5)
#     rules = generateRules(L, suppData, minConf=0.5)
#     print(rules)