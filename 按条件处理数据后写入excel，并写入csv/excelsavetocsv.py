# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:13:51 2020

@author: Gyc
"""

import pandas as pd
import csv
import xlwt

'''
autocsv = r"873-18feature-auto.csv"
releasecsv = r"873-18feature-release.csv"
outcsv = r'out.csv'
'''
df1 = pd.read_csv(r"873-18feature-auto.csv")
df2 = pd.read_csv(r"873-18feature-release.csv")
#print(df1.iloc[:,1].astype(float)/df2.iloc[:,1].astype(float))
#print(type((df1.iloc[:,1].astype(float)/df2.iloc[:,1].astype(float))[0]))

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
# 添加一个sheet，名字为mysheet，参数overwrite就是说可不可以重复写入值，就是当单元格已经非空，你还要写入
sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
col = df2.columns
for a in range(len(col)):
    sheet.write(0,a,col[a])

for i in range(1,df2.shape[1]):
    for j in range(0,len(df1)):
        x=df1.iloc[j,i].astype(float)
        y=df2.iloc[j,i].astype(float)
        ratio=x/y
        #print(ratio)
        if ratio >= 0.5 and ratio <= 2:
            ratio = 1
        else:
            ratio = 0
        sheet.write(j+1,0,df2.iloc[j,0])
        sheet.write(j+1, i, ratio)
        # 记住要保存
book.save(r"873判断2.xls")
outcsv = r'outcsv.csv'

excfile = r"873判断2.xls"
#outcsv = open('outcsv.csv','w',encoding='utf-8')

df3 = pd.read_excel(r"873判断2.xls")
#csvLines = []
with open(outcsv, 'w',newline = "")as f:
    csvWriter = csv.writer(f)
    csvWriter.writerow(col)
    for i in range(len(df3)):
        line = df3.iloc[i,:]
    
        csvWriter.writerows([line])

    
       
'''
郭的方法已成功
for i in range(len(ratio1)):
    if ratio1[i] >= 0.5 and ratio1[i] <= 2:
        ratio1[i] = 1
    else:
        ratio1[i] = 0

with open(outcsv, 'w',newline = "")as f:
    csvWriter = csv.writer(f)
    csvWriter.writerows()
    csvWriter.writerow([ "Number of Nodes(ratio)"])  # 3. 构建列表头
    for i in range(len(ratio1)):
        csv_writer.writerow([ratio1[i]])
'''
'''
段的方法可能成功
csvLines = []
with open(autocsv, 'r')as f:
    csvReader = csv.reader(f)
    for line in csvReader:
        csvLines.append(line)
        csvLines[20] = ratio
        print(line)

with open(outcsv, 'w',newline = "")as f:
    csvWriter = csv.writer(f)
    csvWriter.writerows(csvLines)
'''