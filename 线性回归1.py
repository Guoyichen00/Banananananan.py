# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:07:05 2020

@author: Gyc
"""

#线性回归数据分析代码示例#
from sklearn.model_selection import train_test_split #数据划分相关的类
from sklearn.linear_model import LinearRegression    #线性回归相关的类
from sklearn.preprocessing import StandarScaler      #归一化相关的类

"""导入库"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time

"""设置字符集，防止出现中文乱码"""
mpl.rcParams['front.sans_serif'] = [u'SimHei']
mpl.rcParams['axs,unicode_minus'] = False

"""加载数据"""
path1 = 'data/household_power_constumption_100.txt'
df = pd.read_csv(path1,sep = ':',low_memory = False) #无混合类型时low_memory = False

"""查看数据的前五行信息"""
df.head()

"""查看数据的格式信息"""
df.info()

"""异常数据的过滤"""
new_df = df.replace('?',np.nan)              #替换非法字符为np.nan
datas = new_df.dropna(axis = 0,how = 'any')  #只要有一个数据为nan，就删除整行
datas.describe().T                           #观察数据中的多种统计指标（仅限数值型数据使用）
df.info()                              

"""创建一个时间函数格式化字符串"""
def date_format(dt):
    import time
    t = time.strptime('' , join(dt) , '%d/%m/%Y &H:%M:%S')
    return (t.tm_year,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

"""需求：构建时间和功率之间的映射关系，可以认为：特征属性为时间，目标属性为功率"""
"""获取X和y变量，并将时转换为数值型的连续变量"""
x = data.iloc[:,0:2]
x = x.apply(lambda x : pd_Series(date_format(x),axis = 1))
y = dates['Global_active_power']
x.head(2)

"""对数据进行测试集和训练集的划分"""
'''X：特征矩阵，类型一般为DataFrame'''
'''Y:特征对应的label标签，类型一般是Series'''
'''test_size:对x/y进行划分的时候的测试集数据占比，是一个0-1之间的float数'''
'''random_state：数据分割是基于随机种子分割，该参数制定随机种子，类型为int'''
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)

"""查看训练集上的数据信息"""
x_train.describe()

"""数据标准化"""
'''StandardScaler:将数据转化为标准差为1的数据集'''
'''如果有一个API名字有fit，那么就表示进行模型训练，无返回值'''
'''如果一个API名字中有transform，那么就表示进行数据转换的操作'''
'''如果一个API名字中有predict，那么就表示数据预测，会有一个预测结果输出'''
'''如果API名字中既有fit又有transform，为两者结合，先fit再transform'''
ss = standardScaler()
x_train = ss.fit_transform(x_train)    #训练模型并转换训练集
x_test = ss.transform(x_test)          #利用模型对测试机进行数据标准化
pd.DataFrame(x_train.describe().T)

"""模型训练"""
lr = LinearRegression()
lr.fit(x_train,y_train)           #创建模型b并训练

"""模型校验"""
y_predict = lr.predict(x_test)              #预测结果
print('训练集',lr.score(x_train,y_train))
print('测试集',lr.score(x_test,y_test))
mse = np.average((y_predict - y_test)**2)
rmse = np.sqrt(mse)                         #开方
print('rmse:',rmse)

"""模型保存/持久化"""
'''在机器学习部署中，实际上其中一种方式是将模型进行输出，另一种方式是直接将预测的结果保存'''
'''模型输出值，将模型保存至磁盘文件'''
from sklearn.externals import joblib
joblib.dump(ss,'data_ss.model')           #将标准化模型保存
joblib.dump(lr,'data_lr.model')           #将lr模型保存

"""加载模型"""
ss = joblib.load('data_ss.model')
lr = joblib.load('data_lr.model')

"""作图比较预测值与实际值"""
t = np.arange(len(x_test))
plt.figure(facecolor = 'w')              #穿件一个画布，facecolor设置背景色
plt.plot(t,y_test,'r-',linewidth = 2 ,label = '真实值')
plt.plot(t,y_predict,'g-',linewidth = 2,label = '预测值')
plt.legend(loc = 'upper left')           #显示图例，设置图例位置
plt.title("线性回归",frontsie = 20)
plt.grid(b = true)                       #添加网格
plt.show

"""另外一种方法——最小二乘法求θ值（求得的θ值会忽略截距的影响，不一定是最优解）"""
"""数据导入同上"""
"""数据处理"""
x1 = df.iloc[:,2:4]
y1 = df.iloc[:,5]
x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size = 0.2,random_state = 0)
"""数据矩阵化"""
x2 = np.mat(x1_train)
y2 = np.mat(y_train).reshape(-1,1)
"""计算θ值，θ = （XT*X）-1*XT*Y"""
theta = （x2.T*x2）.I*x2.T*y2
print(theta)
"""对测试集测试"""
y2_hat = np.mat(x_test)*theta
"""作图部分同上“”“
