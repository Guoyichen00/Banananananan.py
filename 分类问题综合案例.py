# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:07:48 2020

@author: Gyc
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings              #发生错误是防止弹出错误窗口
import sklearn 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

"""设置字符集，防止乱码"""
mpl.rcParams['front.sans_serif'] = [u'SimHei']
mpl.rcParams['axs,unicode_minus'] = False

"""拦截异常"""
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

"""加载数据并进行预处理"""
'''加载数据'''
path = "data/crx.data"
names = ['A1','A2','A3']
df = pd.read_csv(path,header=None, names=names) 
print("数据条数：",len(df))

'''异常数据过滤'''
df = df.replace('?',np.nan).dropna(how = 'any')
print("数据过滤后条数：",len(df))

df.head(5)

"""自定义一个哑编码实现方式，将v变量转换成为一个向量/list集合的形式"""
def parse(v,l):
    #V表示一个字符串，代表需要转换的数据类型
    #1是一个类别信息，其中V是l中的一个值,l一般为一个列表
    return[1 if i == v else 0 for i in l]
'''定义一个处理每条数据的函数'''
def parseRecord(record):
    resule = []
    #格式化数据，将离线数据转换为连续数据
    a1 = record['A1']
    for i in parse(a1,('a''b')):
        resule.append(i)
    
    result.append(float(record['A2']))
    result.append(float(record['A3']))

"""数据特征处理（将数据转换为数值类型）"""
new_names = ['A1_0','A1_1','A2_0','A2_1','A2_2']
datas = df.apply(lambda x : pd.series(parseRecord(x),index = new_names),axis = 1)
#对DATAFRAME按照行进行迭代
names = new_names
datas.head(5)

"""数据分割"""
X = datas[names[0:-1]]
Y = datas[names[-1]]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,random_state = 0)
X_train.describe().T

"""数据归一化"""
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
pd.DataFrame(X_train).describe().T

"""逻辑回归模型构建"""
lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50),fit_intercept=True,penalty='l2',solve='lbfgs',tol=0.01,multi_class='ovr')
#Cs表示惩罚项系数的可选则范围（C表示惩罚项的λ），penalty = 'l2'表示使用L2-norm正则
lr.fit(X_train,Y_train)

"""Logistic算法效果输出"""
lr_r = lr.score(X_train,Y_train)
print("Logistic算法在训练集上的R值：",lr_r)
print("Logistic算法稀疏化特征化比率：%2f%%"%(np.mean(lr.coef_.raverl()==0)*100))
print("Logistic算法参数：",lr.coef_)
pirnt("Logistic算法截距:",lr.intercept_)

"""Logistic算法预测（预测所属类别）"""
lr_y_predict = lr.predict(X_test)

"""Logistic算法获取概率值（就是算法计算出来的结果值）"""
y1 = lr.predict_proba(X_test)

"""KNN算法构建"""
knn = KNeighborsClassifier(n_neighbors = 10, algorithm = 'kd_tree',weights = 'distance')
knn.fit(X_train,Y_train)

"""KNN算法效果输出"""
knn_r = knn.score(X_train,Y_train)
print("Logistic算法在训练集上的R值（准确率）:%.2f"%knn_r)

"""KNN算法预测"""
knn_y_predict = knn.predict(X_test)
knn_r_test = knn.score(X_test,Y_test)
print("Logistic算法在测试集上的R值（准确率）:%.2f"%knn_r_test)

"""结果图像显示"""
'''c.图像展示'''
x_len = range(len(X_test))
plt.figure(figsize = (14,7),facecolor = 'w')
plt.ylim(-0,1,1,1)
plt.plot(x_len,Y_test,'ro',markersize = 6,zorder = 3,label = '真实值')
plt.plot(x_len,lr_y_predict,'go',markersize = 10,zorder = 2,label=u'Logis算法预测值')
plt.plot(x_len,knn_y_predict,'yo',markersize = 16,zorder = 1,label=u'KNN算法预测值')
plt.legend(loc = 'center right')
plt.xlabel(u'数据编号',fontsize = 18)
plt.ylabel(u'是否审批（0表示通过，1表示未通过）',fontsize = 18)
plt.title(u'Logistic回归算法和KNN算法对数据进行分类的比较',fontsize = 20)
plt.show()

"""逻辑回归适用于二分类，不适用于多分类"""










