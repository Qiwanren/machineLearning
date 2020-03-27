from sklearn.datasets import make_regression

## 生成特征矩阵，目标向量以及模型的系数
'''
    make_regression参数说明：
        n_samples：样本数
        n_features：特征数(自变量个数)
        n_informative：参与建模特征数
        n_targets：因变量个数
        noise：噪音
        bias：偏差(截距)
        coef：是否输出coef标识
        random_state：随机状态若为固定值则每次产生的数据都一样
    X,Y=make_regression(n_samples=10, n_features=1,n_targets=1,noise=1.5,random_state=1)
    
'''
features , target ,coefficients = make_regression(n_samples=100,
                                                  n_features=3,
                                                  n_informative=3,
                                                  n_targets=1,
                                                  noise=0.0,
                                                  coef=True,
                                                  random_state=1)

### 查看特征矩阵和目标向量
#print('Features Matrix \n',features[:3])
#print('Target Vector\n',target[:3])


X,Y=make_regression(n_samples=10, n_features=1,n_targets=1,noise=1.5,random_state=1)

##  查看数据的长度
print(X.shape)
print(Y.shape)

##  显示点的分布
import matplotlib.pyplot as plt
plt.scatter(
    X, #x坐标
    Y, #y坐标
);
###plt.show()

##  Numpy拟合基于最小二乘法
import numpy as np
#用一次多项式拟合，相当于线性拟合
z1 = np.polyfit(X.reshape(10), Y, 1)
p1 = np.poly1d(z1)
print (z1)
print (p1)

y = z1[0] * X + z1[1]
plt.plot(X, y,c='red')
plt.show()

### 加载库
from sklearn.datasets import make_classification

from sklearn.datasets import make_blobs