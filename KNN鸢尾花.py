 # -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:33:18 2020

@author: Gyc
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import laber_binarize
from sklearn import metrics

"""设置字符集防止乱码"""
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

"""数据加载"""
path = 'data/iris.data'
names =['sepal length','sepal width','petal lengh','petal width','cla']
df = pd.read_csv(path, header = None, names = names)
df['cla'].value_counts()
df.head()

"""数据类型转换"""
def parseRecord(record):
    result = []
    r = zip(names,record)
    for name,v in r:
        if name =='cla'
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
         else:
             result.append(float(v))
    return result

"""数据分割及异常数据清楚"""
'''数据转换'''
datas = df.apply(lamda r:parseRecord(r),axis = 1)
'''异常数据删除'''
datas = datas.dropna(how = 'any')
'''数据分割'''
X = datas[names[0:-1]]
Y = datas[names[-1]]
'''数据抽样'''
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.4,random_state = 0)

"""KNN算法实现"""
'''模型构建'''
knn = KNeighborsClassifier(n_neighbors = 3)   #n_neighbors = 3值模型中的K值为3，n_neighbors用于指定K值
#数据量较大时，使用 knn = KNeighborsClassifier(n_neighbors = 20,algorithm = 'kd_tree',weights = 'distance')语句构建
#其中algorithm指使用什么方式，weights值使用什么作为权重
knn.fit(X_train,Y_train)
'''模型效果输出'''
#将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test,classes = (1,2,3))
#得到预测属于某个类别的概率值
knn_y_score = knn.predict_proba(X_test)
#计算roc的值
knn_fpr,knn_tpr,knn_therasholds = metrics.roc_curve(y_test_hot.ravel(),knn_y_score.ravel())
#计算auc的值
knn_auc = metrics.auc(knn_fpr,knn_tpr)
print("KNN算法R值："，knn.score(X_train,Y_train))
print("KNN算法AUC值：",knn_auc)
#模型预测
knn_y_predict = knn.predict(X_test)

"""画图"""
x_test_len = range(len(X_test))
plt.figure(figsize = (12,9),facecolor = 'w')
plt.ylim(0,5,3,5)
plt.plot(x_test_len,Y_test,'re',markersize = 6, zorder = 3,label = u'真实值')
plt.plot(x_test_len,knn_y_predict,'yo',markersize = 16, zorder = 1,label = u'KNN算法预测值')
ply.legend(loc = 'lower right')
plt.xlabel(u'数据编号',fontsize = 18)
plt.ylabel(u'种类',fontsize = 18)
plt.title(u'鸢尾花数据分类',fontsize = 20)
plt.show()