import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier #GBM算法
from sklearn import model_selection, metrics   #

import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('Train_nyOWmfK.csv')
target = 'Disbursed'
IDcol = 'ID'

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])        #监督学习

    #predict training set
    dtrain_predictions = alg.predict(dtrain[predictors])        #预测样本的标签
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]        #预测为某个标签的概率

    #Perform cross_validation
    if performCV:
        cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')    #计算模型的得分

    #print model report
    print('\nModel Report')
    print('Accuray: %.4g' % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))     #分类准确率
    print('AUC Score(Train):%f' % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))      #根据真实值和预测值计算auc

    if performCV:
        print('CV Scor: Mean - %.7g|MIn - %.7g|Max - %.7g' % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    #Print Feature Importance
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importance')
        plt.ylabel('Feature Importance Score')

#Choose all preditors expect target & IDcols
prefictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, prefictors)