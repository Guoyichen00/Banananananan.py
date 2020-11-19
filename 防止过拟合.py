# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:14:22 2020

@author: Gyc
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticCV
from sklearn.preprocessing import PolynomialFeatures    #数据预处理，标准化
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_decent import ConvergenceWarning

"""设置字符集，防止中文乱码"""
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

"""拦截异常"""
warnings.filterwarnings(action = 'ignore',categroy = ConvergenceWarning)

"""创建模拟数据"""
np.random.seed(100)
np.set_printoptions(linewidth = 1000,suppress = True)   #显示方式设置，设置每行的字符数，用于插入换行符
N = 10
x = np.linspace(0,6,10) + np.random.randn(N)
y = 1.8*x**3 + x**2 - 14*x - 7 + np.random.randn(N)

"""将数据设置为矩阵"""
x.shape = -1,1
y.shape = -1,1

"""构建模型"""
'''RidgeCV和Ridge的区别在与交叉验证，CV表示交叉验证，其他验证模型同理'''
models = [
        Pipeline([
                ('poly',PolynomialFeatures(include_bias = False)),
                ('linear',LinearRegression(fit_intercept = False))
                ]),
        Pipeline([
                ('Poly',PolynomialFeatures(include_bias = False)),
                ('linear'),RidgeCV(alphas = np.logspace(-3,2,50),fit_intercept = False)
                ]),
                #alpha给定的是Ridge算法中，L2正则项的权重，也就是λ
                #alphas是给定CV交叉验证过程中,Ridge算法的alpha参数值的取值范围
        Pipeline([
                ('poly',PolynomialFeatures(include_bias = False)),
                ('linear',LassoCV(alphas = np.logspace(0,1,10),fit_intercept = False))
                ]),
        Pipeline([
                ('Poly',PolynomialFeatures(include_bias = False))
                ('linear',ElasticNetCV(alphas = np.logspace(0,1,10),l1_ratio = [.1,.5,.7,.9,.95,1],
                                        fit_intercept = False))
                #l1_ratio:给定的EN算法中L1正则项在整个惩罚中的比例，这里是给定一个表，表示的是在CV较差验证中，EN算法L1正则项的权重比例的可选范围
                ])
        ]

"""线性回归过拟合图形识别"""
plt.figure(facecolor = 'w')
degree = np.arange(1,N,4)   #阶
dm = degree.size
colors = []
for c in np.linspace(16711680,255,dm):
    colors.append('#%06x',%int(c))
for t in range(4):
    model = models[t]                             #遍历模型
    plt = subplot(2,2,t+1)                        #选择具体子图
    plt.plot(x,y,'ro',ms = 10, zorder = N)        #在子图中画出原始数据点（实际值），zorder表示图像显示在第几层
    for i,d in enumerate(degree):
        '''设置阶数'''
        model.set_params（Poly__degree = d）
        '''训练模型'''
        model.fit(x,y,ravel())
        '''获取得到具体算法模型'''
        lin = model.get_params('Linear')['linear']
        #前一个linear可删除，不过为了兼容不同版本买最好写上对应的名称
        #model.get_params()方法返回的是一个dict对象，后面的Linear是dict对应的key，也就是我们在定义Pipeline时，给定的名称。
        '''打印数据'''
        ouput = u'%s%d阶，系数为：'%(titles[t],d)
        '''判断lin对象中是否有对应属性'''
        if hasattr (lin,'alpha_'):           #判断lin这个模型中是否有alpha属性
            idx = output.find(u'系数')
            output = output[idx] + (u'alpha = %.6f,'%lin.alpha_) + output[idx:]
        if hasattr (lin,'l1_ratio_'):
             idx = output.find(u'系数')
             output = output[idx] + (u'l1_ratio = %.6f,'%lin.l1_ratio_) + output[idx:]
        print(output,lin.coef_.ravel())       #lin.coef_:获取线性模型中的参数列表，即θ值，ravel（）将结果转换
        
        '''产生模拟数据'''
        x_hat = np.linspace(x.min(),x.max(),num = 100)
        x_hat.shape = -1,1
        
        '''预测'''
        y_hat = model.predict(x_hat)
        
        '''计算准确率'''
        s = model.score(x,y)
        z = N-1 if (d==5) else 0   #当d = 5时，设置为N-1层，其他为0层，将d = 5这一层凸显出来
        label = u'%d阶，正确率 = %.3f'%(d,s)
        plt.plot(x_hat,y_hat,color = colors[i],lw = 2,alpha = 0.75,label = label, zorder = z)
        plt.legend(loc = 'upper left')
        plt.grid(True)
        plt.title(titles[t])
        plt.xlabel('x',fontsize = 16)
        plt.ylabel('y',fontsize = 16)
    plt.tight_layout(1,rect = (0,0),0.95)  #图的左右边距
    plt.suptitle(u'不同线性回归过拟合显示',fontsize = 22)
    plt.show