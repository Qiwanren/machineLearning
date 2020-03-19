#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd

data = pd.read_csv('D:/data/python/machine/GameLog.csv',names=['login_id','login_date','login_ip','use_name','work','sex'],encoding='gbk')
### data = pd.read_csv('D:/data/python/machine/GameLog.csv', header=0) 和上面的语句功能相等
## 读取数据后设置列名或者修改列名
# df_example.columns = ['A','B','C']
# 或者在读取的时候，直接指定列名
# df_example = pd.read_csv('Pandas_example_read.csv', names=['A', 'B','C'])
# df_example_noCols = pd.read_csv('Pandas_example_read_withoutCols.csv', header=None, names=['A', 'B','C'])  此处如果使用header=0则会将第一行列名先当为列名，然后被覆盖

## skiprows=1  跳过第一行
print(data.index)  ###  查看data的信息
#print(data.head())### 默认读取5条，不包含标题，默认第一行为标题，也可以传入参数，head(6)  读取前6条

## 设置索引
#data = data.set_index('login_id')      ###  将某个列设置为索引，替换之前自然生成的
#print(data['login_ip'][:5])  ### 只看login_ip列，并且只看前5行

## 设置索引后，可以直接根据索引进行查询
#data1 = data['login_ip']
##data1 = data1['4']   ###  4为索引列的值
##print(data1.max())
#print(data1.min())

#print(data.describe())   ### 对每一个列进行统计和分析

#data1 = data[['login_ip','use_name','sex']]  ####  获取多个列的数据
#print(data1.head(6))

########### 索引操作 #################
###  loc 用label来定位，iloc用postition定位

#data1 = data.iloc[0]  ## 获取第一行数据
#data1 = data.iloc[0:5]  ## 获取前5行数据
#data1 = data.iloc[0:5,1:3]  ## 获取前5行数据 ,列索引从零开始，1:3表示把数据的第一列和第二列拿出来

data1 = data.loc['login_ip','use_name','sex']  ## 定位多个列
print(data1)
#data = data.set_index('use_name')
#print(data.loc['李明克星'])    ###  根据某个索引值进行查询
#print(data.loc['李明克星','sex'])  ### 根据索引定位后，再看具体的某一列的值


##print(data.head(7))

### df == data
###  df.loc[df['Sex'] == 'male','Age'].mean()    #### 取出数据中Sex标签值为male的数据，并求取这些数据中Age字段的平均值
###  (df['Age'] > 70).sum()
#data1 = data[data['login_id'] > 2][:5]   ###  该语句的功能是筛选login_id中大于2的值，并将结果赋值给data1 [:5]表示取前5行
#print(data1)

######
