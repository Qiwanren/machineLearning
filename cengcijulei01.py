import pandas as pd

seeds_df = pd.read_csv('D:/data/python/machine/tjfx/jl/datasets/seeds-less-rows.csv')

varieties = list(seeds_df.pop('grain_variety'))

samples = seeds_df.values

##  距离计算和树状图,hierarch为层次聚类
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt

## 进行层次聚类
mergings = linkage(samples,method='complete')

## 树状图结果
fig = plt.figure(figsize=(10,6))

dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,)

#plt.show()

###  得到结果数据
from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergings,6,criterion='distance')

df = pd.DataFrame({'labels':labels,'varieties':varieties})
ct = pd.crosstab(df['labels'],df['varieties'])

print(ct)