# code=utf-8
import pandas as pd
import re

# 打开文件
'''
    打开文件，将字符串中数字提取出来，以list的格式，新增一列
    data = pd.read_csv(r'D:/data/python/work/cb_product_info_20200715.txt',sep=',',names=['PRODUCT_ID','PRODUCT_NAME','PRODUCT_ZF','START_DATE','END_DATE','IS_CARD'])
data['PRODUCT_ZF1'] = data['PRODUCT_ZF'].str.extract('(\d+)')
data['PRODUCT_ZF2'] = data['PRODUCT_NAME'].str.findall('(\d+)元').tolist()
data.to_csv('D:/data/python/work/cb_product_info_20200715_4444.csv')

'''
data = pd.read_csv(r'D:/data/python/work/cb_product_info_20200715_4444.csv',sep=',',names=['PRODUCT_ID','PRODUCT_NAME','PRODUCT_ZF','START_DATE','END_DATE','IS_CARD','PRODUCT_ZF1','PRODUCT_ZF2'])

def function(value):
    list1 = value.split(',')
    if len(list1) > 0:
        return re.sub("\D", "", list1[-1])
    else:
        return ''

data['PRODUCT_ZF3'] = data.apply(lambda x: function(x.PRODUCT_ZF2), axis = 1)

data.to_csv('D:/data/python/work/cb_product_info_20200715_555.csv')

##frame['test'] = frame.apply(lambda x: function(x.city, x.year), axis = 1)
