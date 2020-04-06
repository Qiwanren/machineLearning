from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
import numpy as np

#生成二分类的isotropic Gaussian blobs
X,y = make_blobs(centers=2)
# 生成数据集
mglearn.discrete_scatter(X[:,0],X[:,1],y)
# 图像展示
plt.legend(['Class 0','Class 1'],loc=4)
#打印出来生成的点的个数看一看2333
print(X.shape,y.shape)
print({n:v for n,v in zip(['Class 0','Class 1'],np.bincount(y))})
plt.show()




