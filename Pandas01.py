#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

data = pd.read_csv('D:/data/python/machine/t_base_station_info.csv',names=['id','prov_code','city_code','lac','net_type','node_type','status'],encoding='gbk')

#print(data.info(memory_usage='deep'))

#print(data.iloc[6:8])

#data1 = data['prov_code']   ### 选取一个列
#cols = ['prov_code','city_code','lac','net_type']   ####  选取多个列
#data1 = data[cols]

##criteria = (data['net_type'] == 2) & (data['node_type'] == 2)  ### 获取net_type == 2 并且 node_type == ‘2’的数据
##print(data[criteria].head())                                   ### 显示数据

##criteria = (data['net_type'] < 2) | (data['node_type'] > 2)  ### 获取net_type == 2 并且 node_type == ‘2’的数据
##print(data[criteria].head())                                   ### 显示数据

##criteria = data['name'].str.contains('White')  ### 获取name列值内容中包含White内容的
###data[criteria].head()

###  data['is_selected'] = data['node_type'] > 2    ####  新增一个列
###  print(data[6:9])

# 将 date 列转换成 datetime 类型

##  df['date'] = pd.to_datetime(df['date'])      ####  将date数据转换为日期类型
# 筛选条件为日期大于 2014/4/1
###  criteria = df['date'] > pd.Timestamp(2014,4,1)   ####
#####  df[criteria].head()

###  同时选择行和列
#where = data['net_type'] > 2              ###  先选择行
#cols = ['prov_code','city_code','lac','net_type']     ###  再选择列
###print(data[where][cols].head(5))

### -------------- group by 的使用方式 ----------------------------------------

#df = pd.DataFrame({'key':['A','B','C','A','B','C','A','B','C'],'data':[0,5,10,5,10,15,10,15,20],'data1':[0,15,10,5,15,15,10,15,20]})
#print(df.head())

#print(df.groupby('key').sum())   ### 对key分组求和，如果不指定列，则对所有列求和
#print(df.groupby('key').data.sum())   ### 对key分组求和，对指定列求和，mean求取平均值
#print(df.groupby('key').data.describe())  ####  分组后对应列的相关情况
#print(df.groupby('key')['data'].sum())  ####  分组后对应列的相关情况

####  创建一个新的dataFrame
#df1 = pd.DataFrame([[1,2,3],[4,5,6]],index=['aa','bb'],columns=['A','B','C'])   ###  指定行，索引列，列名称
#print(df1.sum(axis=0))   ###  按照列进行求和
#print(df1.sum(axis=1))   ###  按照行进行求和  平均值 mean, 最大最小值，中位数 median()，都可以

###########  二元统计
'''
   计算数字之间的相关性 使用corr方法
   检查两个变量之间变化趋势的方向以及程度，值范围-1到+1，0表示两个变量不相关，正值表示正相关，负值表示负相关，值越大相关性越强
    corr 实质上是对其每一列数据进行相关系数的计算，其结果等同于取出每列数据采用 np.corrcoef 计算
    
    pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
    kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
    spearman：非线性的，非正太分析的数据的相关系数

'''
#print(df1.corr())   ####  corr方法显示各个列之间的相关性，
# 计算第一列和第二列的相关系数
###print(data.one.corr(data.two))   显示第一列和第二列直接的相关性，one 和 two为列名

#df = pd.DataFrame({'A':np.random.randint(1, 100, 10),
#                     'B':np.random.randint(1, 100, 10),
#                     'C':np.random.randint(1, 100, 10)})

#print(df.corr())  # 计算pearson相关系数
#print(df.corr('kendall'))      # Kendall Tau相关系数
#print(df.corr('spearman'))     # spearman秩相关


###  使用value_counts()  计算某一列中所有值出现的次数
##print(data['net_type'].value_counts(ascending=True))  ## 指定统计后的值按升序排列，默认是降序排列的，当net_type的值比较多的时候，可以使用bins = 5 ,将结果等差的分为5个区间，看每个区间有多少
###print(data['net_type'].count())   ##  统计列中有多少条数据
'''
    Series学习开始
'''
data1 = ['10','11','12']
index1 = ['a','b','c']
#s = pd.Series(data1,index = index1)
#print(s)

#s1 = s.copy()
#s1['a'] = 100    ####   直接对某个元素修改
#1['g'] = 401    ###    在s1中新增一行数据
#print(s1[1:2])
#s2 = s1.replace(to_replace = 100,value= 101,inplace = False)   ###  对某个元素进行修改，inplace = False 的时候，对于对象本身不做修改，只是将修改结果展示
#print(s2)
#s1.replace(to_replace = 100,value= 101,inplace = True)
#print(s1)

#s.index = ['a','b','d']   ### 用一个新的序列替换原来的索引
#print(s)

s2 = pd.Series([100,500],index=['g','h'])
#print(s2)

#s3 = s1.append(s2,ignore_index = False)  ##  拼接序列,形成一个新的对象，s1本身是不变的,ignore_index参数指定是否忽略原来的索引
#print(s1)
#print(s3)

#del s1['a']    ### 删除索引为a的行

#s1.drop(['b','c'],inplace = True)   ###  一次删除多个列,如果没有对应的索引，会提示异常

#print(s1)

'''
    merge操作
    pandas中的merge()函数类似于SQL中join的用法，可以将不同数据集依照某些字段（属性）进行合并操作，得到一个新的数据集
    DataFrame1.merge(DataFrame2, how=‘inner’, on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=(’_x’, ‘_y’))
    #Right链接
    dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey',how='right')
    #Outer链接
    dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='outer')
'''
left = pd.DataFrame({'A':['A0','A1','A4'],
                     'B':['B0','B1','B2'],
                     'C':['C0','C1','C2'],
                     'H':['C0', 'C1', 'C2']})

right = pd.DataFrame({'A':['A0','A1','A5'],
                     'D':['D0','D1','D2'],
                     'G':['G0','G1','G2']})

#print(left)
#print(right)

## merge_data = pd.merge(left,right)   ###  left和right合并，默认是根据相同的列进行合并的
#merge_data = pd.merge(left,right,on='A')   ###  left和right合并，默认是根据相同的列进行合并的，默认为内连接  on = ['A','B']  多个列进行连接
#print(merge_data)

'''
    数据透视表
        透视表就是将指定原有DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数(默认情况下式mean函数)。
        
        可以使用pivot和pivot_table
        pivot要求：在行与列的交叉点值的索引应该是唯一值，如果不是唯一值，则会报错
        ({'foo': ['one','one','one','one','two','two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'baz': [1, 2, 3, 4, 5, 6]})
        则索引值： 1和4 所对应的foo 和 bar是同一个值，但在原序列中的索引值不相同
        pivot_table没有特殊的限制要求，当索引值不相等的时候，会合并
    数据交叉表（频率可以看作出现次数）
        交叉表是用于统计分组频率的特殊透视表
        print(pd.crosstab(df['类别'],df['产地'],margins=True)) # 按类别分组，统计各个分组中产地的频数（出现次数）
'''
df_test = pd.DataFrame({'foo': ['one','one','one','two','two','two','one','one','one','two','two','two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C','A', 'B', 'C', 'A', 'B', 'C'],
                       'baz': [1, 2, 3, 4, 5, 6,9,7,8,4,5,6]})
###df1 = df_test.pivot(index='foo', columns='bar', values='baz')
df1 = df_test.pivot_table(index='foo', columns='bar', values='baz')   ###  bar为列表头, 默认求baz的平均值
#df1 = df_test.pivot_table(index='foo', columns='bar', values='baz'，aggfunc='max')  求最大值，当aggfunc='count'表示计数
#print(df1)
#print(df1.sum(axis=1))   ###  对透视图的结果按行求和
#print(df1.sum(axis=0))   ###  对透视图的结果按列求和

'''
    日期时间操作
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time')
        data[pd.Timestamp('2012-01-01 90:00'):pd.Timestamp('2012-01-01 19:00')]  ## 取出日期区间中的数据，时间切片
        
'''

###  获取列的唯一值
data = pd.DataFrame(['a','a','b','c','d','d','d','d','d','e','e','e','e'],columns=['X'])
df_list = data['X'].unique()
print(df_list)




















