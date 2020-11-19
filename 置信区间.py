# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:19:01 2020

@author: Gyc
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '测试.xlsx'

df1 = pd.read_excel(path)
df3 = df1.iloc[:,:]

'''def average(data):
    return sum(data)/len(data)'''


def bootstrap(data,B,c):
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0,n,size = n)
        data_sample = array[index_arr]
        sample_result = data_sample.mean()
        sample_result_arr.append(sample_result)
        
    a = 1-c
    K1 = int(B*a/2)
    K2 = int(B*(1-a/2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[K1]
    higher = auc_sample_arr_sorted[K2]
    
    
    return lower,higher #,auc_sample_arr_sorted

'''
df2 = df3.iloc[:,3]
low_bon,high_bon,sample_arr = bootstrap(df2,1000,0.95)
df3 = df3[(low_bon < df2)]  

df4 = sorted(df2)
x1 = range(1,len(sample_arr)+1)
y1 = sample_arr

plt.plot(x1,y1)
plt.show()
'''
for i in range(3,9):
    low_bon,high_bon = bootstrap(df1.iloc[:,i],1000,0.9)
    df3 = df3[(low_bon < df3.iloc[:,i]) & (df3.iloc[:,i] < high_bon)]    
    print(low_bon,high_bon)
print(df3)