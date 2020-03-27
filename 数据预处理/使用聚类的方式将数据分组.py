import pandas as pd

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans


#创建模拟的特征矩阵
features , _ = make_blobs(n_samples=50,
                          n_features=2,
                          centers=3,
                          random_state=1)
print(features)