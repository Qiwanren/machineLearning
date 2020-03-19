#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

data = pd.DataFrame({'key':['a','a','a','b','b','b','c','c','c'],
                     'data':['2','3','4','5','3','6','7','8','9']})

data.sort_values(by=['key','data'],ascending = [False,True],inplace = True)   ###  将key,data两列进行排序，第一列降序，第二列升序，inplace= True表示将该操作在源数据生效
#print(data)

#data.drop_duplicates()  ### 去除重复的数据
#data.drop_duplicates(subset='key')  ### 以key为主，去除重复的数据

data1 = pd.DataFrame({'data':['1','2','3','4','5','6','7'],
                      'food':['A1','A2','B1','B2','B3','C1','C2']})

def food_map(series):
    if series['food'] == 'A1':
        return 'A'
    elif series['food'] == 'A2':
        return 'A'
    elif series['food'] == 'B1':
        return 'B'
    elif series['food'] == 'B2':
        return 'B'
    elif series['food'] == 'B3':
        return 'B'
    elif series['food'] == 'C1':
        return 'C'
    elif series['food'] == 'C2':
        return 'C'

##data1['food_map'] = data1.apply(food_map,axis = 'columns')  ###  新增一个列

#print(data1)

foodMap = {
    'A1':'A',
    'A2':'A',
    'B1':'B',
    'B2':'B',
    'B3':'B',
    'C1':'C',
    'C2':'C'
}

data1['food_map1'] = data1['food'].map(foodMap)
#print(data1)
'''
    分组和聚合
        分组：groupBy  聚合：agg
    groupBy 升级
    print ("groupby 后，每组只抽样一个")
    print (df.groupby("group").head(1),"\n")
    
    count()：统计出现的次数
    df_agg = dff1.groupby('Country').agg(['min', 'mean', 'max'])
    如果只是指定分组列，则对其他所有剩余的列采用聚合算法
    
'''
dff = pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','foo'],
                    'B':['one','one','two','three','two','two','one','three'],
                    'C': np.random.randn(8),
                    'D':np.random.randn(8)})

#grouped = dff['C'].groupby(dff['A']).mean()  ### 将A作为分组键值，对C进行分组，再求每组的均值
### grouped = dff['C'].groupby([dff['A'],dff['B']]).mean()
#grouped = dff.groupby(dff['A']).count().agg(get_letter_type)
#print(grouped)

def get_letter_type(letter):
    print('this is : ')
    print(letter)

dff1 = pd.DataFrame({'Country':['China','China', 'India', 'India', 'America', 'Japan', 'China', 'India'],
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],
                    'Age':[5000, 4321, 1234, 4010, 250, 250, 4500, 4321]})

#print(dff1.head())
df_gb = dff1.groupby('Country').count()   ###
df_agg = dff1.groupby('Country').agg(['min', 'mean', 'max'])
df_agg1 = dff1['Income'].groupby(dff1['Country']).agg(['min', 'mean', 'max'])  ###  指定分组列和统计列
num_agg = {'Age':['min', 'mean', 'max']}
#print(dff1.groupby('Country').agg(num_agg))   ####  统计结果和上面的相同
num_agg = {'Age':['min', 'mean', 'max'], 'Income':['min', 'max']}
#print(dff1.groupby('Country').agg(num_agg))


#print(dff1)
#df_agg2 = dff1['Income'].groupby(dff1['Country']).agg(get_letter_type)  ###  指定分组列和统计列，get_letter_type方法中参数接受到是Income值和对应的索引
num_agg = {'Age':['min', 'mean', 'max']}
#print(dff1.groupby('Country').agg(num_agg))
#print(df_agg2)
###print(dff1.groupby('Country').size())  ##  返回分组大小


'''
    pandas绘图
'''
import matplotlib.pyplot as plt

s = pd.Series(np.random.randn(10),index=np.arange(0,100,10))
#print(s)
#s.plot()
#第二步利用pandas的plot方法绘制折线图
#df.plot(x = "Year", y = "Agriculture")
#第三步: 通过plt的show()方法展示所绘制图形
#plt.show()

#df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
 #                 index = np.arange(0,100,10),
#                columns = ['A','B','C','D'])
#print(df.head())



