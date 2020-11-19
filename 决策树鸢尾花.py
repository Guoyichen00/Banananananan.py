# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:17:53 2020

@author: Gyc
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn import tree #决策树
from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.model_selection import train_test_split #测试集与训练集划分
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量

from sklearn.preprocession import MinMaxScaler #数据归一化
from sklearn.decomposition import PCA #主成分分析
from sklearn.model_selection import GridSearchCV #网格搜索交叉验证，用于选择最优的参数

"""设置中文乱码"""
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus']  = False

"""拦截异常信息"""
warnings.filterwarnings('ignore',category = FutureWarning)

iris_feature_E = 'sepal length','sepal width','petal length','petal width'
iris_feature_C = '花萼长度','花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa','Iris-versicolor','Iris-virginica'

"""读取数据"""
path = './datas/iris.data'
data = pd.read_csv(path,header = None)
x = data[list(range(4))]   #获取X变量
y = pd.Categorical(data[4]).codes  #把Y转换成分类型的0,1,2
print("总样本数目：%d;特征属性数目：%d" % x.shape)

"""对数据进行分割"""
x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,train_size = 0.8,random_state = 0)
x_train,x_test,y_train,y_test = x_train1,x_test1,y_train1,y_test1
print("训练数据集样本数目：%d,册数数据集样本数目：%d"%(x_train.shape[0],x_test.shape[1]))
'''y强制转换为int类型，因为DecisionTreeClassifier是分类算法，要求y必须是int类型'''
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

"""数据预处理的常用方法"""
    """数据标准化"""
    '''StandardScaler(基于特征矩阵的列，将属性值转换至服从正态分布)'''
    '''标准化是依照特征矩阵的列处理数据,其通过z-score的方法，将样本的特征值转换到同一量纲下'''
    '''常用于基于正态分布的算法，比如回归'''

    """数据归一化"""
    '''MinMaxScaler(区间缩放，基于最大最小值，将数据转换到0,1区间上)'''
    '''提升模型收敛速度，提升模型精度'''
    '''常见于神经网络'''

    """Normalizer(基于矩阵的行，将样本向量转换为单位向量)"""
    '''目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准'''
    '''常见用于文本分类和聚类，logistic回归中也会使用，有效防止过拟合'''
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print("原始数据各个特征属性的调整最小值：",ss.min_)
print("原始数据各个特征属性的缩放调整值：",ss.scale_)

"""方法1："""
"""特征选择：从已有的特征中选择出影响目标值最大的特征属性"""
'''常用方法： {分类：F统计量，卡方系数，互信息mutual_info_classif}，
            {连续：皮尔逊相关系数，F统计量，互信息mutual_info_classif}'''

'''SelectKBest(卡方系数)'''
ch2 = selectKBest(chi2,k =3) #在当前案例中，使用SelectKBest这个方法从4个原始的特征属性，选择3个影响最大的特征属性
        #K默认为10
        #如果指定了，就会返回你所想要的特征个数
x_train = ch2.fit_transform(x_train,y_train)
x_test = ch2.transform(x_test)  #转换

select_name_index = ch2.get_support(indices = True)
print("对类别判断影响最大的三个特征属性分别是：",ch2.get_support(indices = False))
print(select_name_index)

"""方法2："""#推荐
"""降维:对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高阶）数据变成低维数据进行处理"""
#降维可以理解为将影响度小的特征属性融合如影响度打的特征属性之中，从而实现特征属性的减少
'''常用方法： PCA:主成分分析（无监督）
            LDA：线性判别分析（有监督）类内方差最小，人脸识别通常先做一次PCA'''
pca = PCA(n_components = 2) #设置一个PCA对象，设置最终维度为2维(此处是为了画图方便设置为2维)，通常使用默认值，不设置参数
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

"""模型构建"""
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)#criterion可以选择gini
#参数criterion
#参数splitter:选定划分方式，默认为best，即依据训练集的情况选定最佳方式，可能会有过拟合现象，可换成random
#参数max_features:选择选取的特征数，可输入int，float，auto等，int表示选取多少个特征值，float表示选取多少比例的特征值，auto自动选择。
#参数max_depth:表示树的深度
model.fit(x_train,y_train)
y_test_hat = model.predict(x_test)

"""模型结果评估"""
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_test_hat)
print("准确率：%.2f%%"%(np.mean(result)*100))
#实际可通过参数获取
print("Score:",model.score(x_test,y_test))#准确率

"""画图"""
N = 100 #横纵坐标各取多少个值
x1_min = np.min((x_train.T[0].min(),x_test.T[0].min()))
x1_max = np.max((x_train.T[0].max(),x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(),x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(),x_test.T[1].max()))

t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_min,N)
x1,x2 = np.meshgrid(t1,t2)  #生成网络采样点
x_show = np.dstack((x1.flat,x2.flat))[0]#测试点

y_show_hat = model.predict(x_show)#预测值

y_show_hat = y_show_hat.reshape(x1.shape) #使之与输入的形状相同
print(y_show_hat.shape)
print(y_show_hat[0])

'''设置区域划分是的背景颜色'''
plt_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g','r','b'])

plt.figure(facecolor='w')

'''画一个区域'''
plt.pcolormesh(x1, x2,y_show_hat,camp = plt_light)
'''画测试数据点的信息'''
plt.scatter(x_test.T[0],x_test.T[1],c = y_train.ravel(),
            edgecolors = 'k',s =150,zorder = 10,cmap = plt_dark,marker = '*')
#测试数据
'''画训练数据点的信息'''
plt.scatter(x_train.T[0],x_train.T[1],c = y_train.ravel(),
            edgecolors = 'k',s =40,cmap =plt_dark)#全部数据
plt.xlabel(u'特征属性1',fontsize = 15)
plt.ylaber(u'特征属性2',fontsize = 15)
plt.xlim(x1_min,x1_max) #X坐标轴范围
plt.ylim(x2_min,x2_max) #Y坐标轴范围
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类',fontsize = 18)
plt.show()

"""参数优化"""
pipe = pipeline([('mms',MinMaxScaler()),   #最大最小的选择
                 ('skb',SelectKBest(chi2)),#参数选择
                 ('pca',PCA()),            #降维 
                 ('decision',DecisionTreeClassifier(random_state = 0))])#构建决策树
'''参数给与'''
parameters = ('skb__k':[1,2,3,4]
              'pca__n_components':[0.5,0.99]
              #设置位浮点数代表主成分方差所占最小比例的阈值,也就是降维之后所剩余的
              #维度是多少，如100个维度，参数为0.5，则降维后有50个维度
              #如果是int型，则是指定降维后剩的维度
              'decision__criterion':['gini','entropy']
              'decision__max_depth':[1,2,3,4,5,6,7,8,9,10])
'''数据'''
x_train2,x_test2,y_train2,y_test2 = x_train1,x_test1,y_train1,y_test1
'''模型构建：通过网格交叉验证，寻找最优参数列表，param_grid为可选参数列表，cv:进行几折交叉验证'''
gscv = GridSearchCV(pipe,param_grid = parameters,cv =3)
'''模型训练'''
gscv.fit(x_train2,y_train2)
'''算法最优解'''
print('最优参数列表:',gscv.best_params)
print('Score值:',gscv.best_score_)
print('最优模型:',end = '')
print(gscv.best_estimator_)
'''预测值'''
y_test_hat2 = gscv.predict(x_test2)