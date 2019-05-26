'''
朴素贝叶斯
'''
import numpy as np
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
import re
import random
import os
import jieba
import matplotlib.pyplot as plt
'''
创建实验样本

'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],          #切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]        #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec     #实验样本切分的词条，类标签向量

# postingList,classVec = loadDataSet()
# for each in postingList:
#     print(each)
# print(classVec)

'''
根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

'''
def setOfWords2Vec(vocabList, inputSet):        #creatVocaList返回的列表；切分的词条列表
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec        #文档向量，词集模型

'''
将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

'''
def createVocabList(dataSet):       #整理的样本数据集
    vocabSet = set([])      #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)     #取并集
    return list(vocabSet)       #返回不重复的词条列表，也就是词汇表

# postingList, classVec = loadDataSet()
# print('postingList:\n', postingList)
# myVocabList = createVocabList(postingList)
# print('myVocabList:\n', myVocabList)
# trainMat = []
# for postinDoc in postingList:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# print('trainMat:\n', trainMat)

'''
朴素贝叶斯分类器训练函数

'''
def trainNB0(trainMatrix, trainCategory):
    #trainMatrix-训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵；
    #trainCategory-训练类别标签向量，即loadDataSet返回的classVec
    numTrainDocs = len(trainMatrix)     #计算训练的文档数目
    numWords = len(trainMatrix[0])      #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #创建numpy.zeros数组，词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0        #分母初始化为2，拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:       #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:       #统计属于非侮辱类的条件概率所系的数据，即P(w0|0),P(w1|0),P(w2|0)...
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)      #取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive       #返回属于侮辱类的条件概率数组；属于非侮辱类的条件概率数组；文档属于侮辱类的概率


# postingList,classVec = loadDataSet()
# myVocabList = createVocabList(postingList)
# print('myVocabList:\n', myVocabList)
# trainMat = []
# for postinDoc in postingList:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNB0(trainMat, classVec)
# print('p0V:\n', p0V) #p0V存放的是每个单词属于类别0，也就是非侮辱类词汇的概率
# print('p1V:\n', p1V) #p1V存放的是每个单词属于侮辱类的条件概率
# print('classVec:\n',classVec)
# print('pAb:\n', pAb) #pAb存放的是所有侮辱类的样本占所有样本的概率

'''
朴素贝叶斯分类器分类函数
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #vec2Classify - 待分类的词条数组
    #p0Vec - 非侮辱类的条件概率数组
    #pClass1 - 文档属于侮辱类的概率
    # p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1 #对应元素相乘
    # p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    # print('p0:',p0)
    # print('p1:',p1)
    if p1 > p0:
        return 1 #属于侮辱类
    else:
        return 0 #属于非侮辱类

'''
测试朴素贝叶斯分类器
'''
def testingNB():
    listOPosts,listClasses = loadDataSet()      #创建实验样本
    myVocabList = createVocabList(listOPosts)       #创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))     #将实验样本向量化
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))        #训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']     #训练样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))      #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')       #执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']       #测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))      #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')       #执行分类并打印分类结果

# if __name__ == '__main__':
#     testingNB()


'''
过滤垃圾邮件
'''
'''
接收一个大字符串，并将其解析为字符串列表
'''
def textParse(bigString):       #将字符串转换为字符列表
    listOfTokens = re.split(r'\W+',bigString)       #将特殊符合作为切分标志进行字符串切分，即非字母、非数字
    a = [tok.lower() for tok in listOfTokens if len(tok) > 2]        #除了单个字母，例如大写的I，其他单词变成小写
    return a

# if __name__ == '__main__':
#     docList = []; classList = []
#     for i in range(1, 25):      #遍历25个txt文件
#         # try:
#         #     wordList = open('/Users/user/Desktop/Machine-Learning-master/Naive Bayes/email/ham/%d.txt' % i,
#         #                     encoding='UTF-8').read()
#         # except UnicodeDecodeError:
#         #     print(i)
#         # finally:
#         #     print(wordList)
#         wordList = textParse(open('/Users/user/Desktop/Machine-Learning-master/Naive Bayes/email/spam/%d.txt' % i, 'r').read())
#         #读取每个垃圾邮件，并将字符串转换成字符串列表
#         docList.append(wordList)
#         classList.append(1)     #标记垃圾邮件，1表示垃圾文件
#         wordList = textParse(open('/Users/user/Desktop/Machine-Learning-master/Naive Bayes/email/ham/%d.txt' % i, 'r').read())
#         #读取每个非垃圾文件，并将字符串转换成字符串列表
#         docList.append(wordList)
#         classList.append(0)     #标记非垃圾邮件
#     vocabList = createVocabList(docList)        #创建词汇表，不重复
#     print (len(classList))
#     print(vocabList)

'''
测试朴素贝叶斯分类器
'''
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 25):      #遍历25个txt文件
        wordList = textParse(open('/Users/user/Desktop/Machine-Learning-master/Naive Bayes/email/spam/%d.txt' % i).read())
        #读取每个垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)     #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('/Users/user/Desktop/Machine-Learning-master/Naive Bayes/email/ham/%d.txt' % i).read())
        #读取每个非垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)        #创建词汇表，不重复
    trainingSet = list(range(48)); testSet = []     #创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):     #从50个邮件中，随机挑选出40个作为训练集，10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))        #随机选取索引值
        testSet.append(trainingSet[randIndex])      #添加测试集的索引值
        del(trainingSet[randIndex])     #在训练集列表中删除添加到测试集的索引值
    trainMat = []; trainClasses = []        #创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:        #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])        #将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))      #训练朴素贝叶斯模型
    errorCount = 0      #错误分类计数
    for docIndex in testSet:        #遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])       #测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:        #如果分类错误
            errorCount += 1     #错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))

# if __name__ == '__main__':
#     spamTest()

'''
新浪新闻分类
'''
'''
中文文本处理
'''
def TextProcessing(folder_path, test_size = 0.2):
    #folder_path 文本存放的路径
    #test_size 测试集占比，默认占所有数据集的百分之20
    os_list = os.listdir(folder_path)       #查看folder_path下的文件
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(folder_path, item)):        #用于判断某一对象（需提供绝对路径）是否为文件
            os_list.remove(item)
    folder_list = os_list
    data_list = []      #数据集数据
    class_list = []     #数据集类别
    #os.path.isdir()和os.path.isfile()需要传入的参数是绝对路径，但是os.listdir()返回的只是一个某个路径下的文件和列表的名称.
    #遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)     #路径拼接，根据子文件夹生成新的路径
        files = os.listdir(new_folder_path)     #存放子文件夹下的txt文件列表

        j = 1
        #遍历每个txt文件
        for file in files:
            if j > 100:     #每类txt样本书最多100
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:       #打开txt文件
                raw = f.read()
            #需要分词的字符串；cut_all参数用来控制是否采用全模式；
            word_cut = jieba.cut(raw, cut_all = False)      #精简模式，返回一个可以迭代的generator
            word_list = list(word_cut)      #generator转换为list
            data_list.append(word_list)     #添加数据集数据
            class_list.append(folder)       #添加数据集类别
            j += 1
        # print(data_list)
        # print(class_list)
    data_class_list = list(zip(data_list, class_list))      #zip压缩合并，并将数据与标签对应压缩
    random.shuffle(data_class_list)     #将data_class_list乱序
    # print(len(data_class_list))
    index = int(len(data_class_list) * test_size) + 1       #训练集和测试集切分的索引值
    train_list = data_class_list[index:]        #训练集
    test_list = data_class_list[:index]     #测试集
    train_data_list, train_class_list = zip(*train_list)       #训练集压缩
    test_data_list, test_class_list = zip(*test_list)       #测试集压缩

    all_words_dict = {}     #统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)  #根据键值倒序排列
    all_words_list, all_words_nums = zip(*all_words_tuple_list) #解压缩
    all_words_list = list(all_words_list)       #转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

# if __name__ == '__main__':
#     folder_path = '/Users/user/Desktop/Machine-Learning-master/Naive Bayes/SogouC/Sample'
#     # TextProcessing(folder_path)
#     all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
#     print(all_words_list)

'''
根据feature_words将文本向量化
'''
def TextFeatures(train_data_list, test_data_list, feature_words):
    #train_data_list 训练集；test_data_list 测试集；feature_words 特征集
    def text_feature(text, feature_words):      #出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_feature(text, feature_words) for text in train_data_list]
    text_feature_list = [text_feature(text, feature_words) for text in test_data_list]
    return train_feature_list, text_feature_list

'''
读取文件里的内容，并去重
'''
def MakeWordsSet(words_file):
    #文件路径
    words_set = set()       #创建set集合
    with open(words_file, 'r', encoding='utf-8')as f:       #打开文件
        for line in f.readlines():      #一行一行读取
            word = line.strip()     #去回车
            if len(word) > 0:       #有文本，则添加到words_set中
                words_set.add(word)
    return words_set        #读取的内容的set集合

'''
文本特征选取
'''
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    #all_words_list 训练集所有文本列表；deleteN 删除词频最高的deleteN个词；stopwords_set 指定的结束语
    feature_words = []      #特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:        #feature_words的维度为1000
            break
        #如果这个词不是数字，并且不是指定的结束语，单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

# if __name__ == '__main__':
#     folder_path = '/Users/user/Desktop/Machine-Learning-master/Naive Bayes/SogouC/Sample'
#     all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
#
#     stopwords_file = '/Users/user/Desktop/Machine-Learning-master/Naive Bayes/stopwords_cn.txt'
#     stopwords_set = MakeWordsSet(stopwords_file)
#
#     feature_words = words_dict(all_words_list, 100, stopwords_set)
#     print(feature_words)

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

# if __name__ == '__main__':
#     folder_path = '/Users/user/Desktop/Machine-Learning-master/Naive Bayes/SogouC/Sample'
#     all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
#     stopwords_file = '/Users/user/Desktop/Machine-Learning-master/Naive Bayes/stopwords_cn.txt'
#     stopwords_set = MakeWordsSet(stopwords_file)
#     # test_accuracy_list = []
#     # deleteNs = range(0, 1000, 20)
#     # for deleteN in deleteNs:
#     #     feature_words = words_dict(all_words_list, deleteN, stopwords_set)
#     #     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
#     #     test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
#     #     test_accuracy_list.append(test_accuracy)
#     #
#     # plt.figure()
#     # plt.plot(deleteNs, test_accuracy_list)
#     # plt.title('Relationship of deleteNs and test_accuracy')
#     # plt.xlabel('deleteNs')
#     # plt.ylabel('test_accuracy')
#     # plt.show()
#     test_accuracy_list = []
#     feature_words = words_dict(all_words_list, 450, stopwords_set)
#     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
#     test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
#     test_accuracy_list.append(test_accuracy)
#     ave = lambda c:sum(c) / len(c)
#
#     print(ave(test_accuracy_list))