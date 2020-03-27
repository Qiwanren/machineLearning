'''
    原理：根据某个条件对数据二分类

    对于二值化操作：使用两种方法
        第一种方法：
             求出大于等于1的索引值，令这些索引值对应的数值等于1，然后重新构建列
        第二种方法：
            使用Binarizer(threshold=0.9) 表示大于0.9的数据使用1表示
            这里传入的参数需要是二维的，因此需要做维度转换
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.style.reload_library()
plt.style.use('classic')
# 设置颜色
mpl.rcParams['figure.facecolor'] = (1, 1, 1, 0)
# 设置图形大小
mpl.rcParams['figure.figsize'] = (6.0, 4.0)
# 设置图形的分辨率
mpl.rcParams['figure.dpi'] = 100

popsong_df = pd.read_csv('datasets/song_views.csv', encoding='utf-8')
# 我们对listen_count听歌的次数进行二值化操作, 听过的次数大于等于1的为1，次数为0的为0
# 第一种方法
# listened = popsong_df['listen_count'].copy()
# listened[listened >= 1] = 1
# popsong_df['listened'] = listened
# print(popsong_df[['listen_count', 'listened']])

# 第二种方法：使用 Binarizer
from sklearn.preprocessing import Binarizer  # threshold表示阈值，大于0.9的为1
bin = Binarizer(threshold=0.9)
popsong_df['listened'] = bin.transform(popsong_df['listen_count'].values.reshape(-1, 1))   ## reshape来更改数据的列数和行数
print(popsong_df[['listen_count', 'listened']].iloc[:10])