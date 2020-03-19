#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample = 20
x = np.linspace(0,10,nsample)    ###  从零到十之间，选取nsample个数
X=sm.add_constant(x)

### 设置 β0 和  β1 的值分别为2,5

beta = np.array([2,5])
print(beta)

### 误差项
e = np.random.normal(size=nsample)

## 实际值 y
y = np.dot(X,beta) + e

###  最小二乘法
model = sm.OLS(y,X)

###  拟合数据
res = model.fit()

###  回归系数
parms = res.params    ### 此时求解出的是实际的的 β0 和  β1

###  显示函数的相关信息，用于函数效果评估
res.summary()

#### 拟合的估计值
y_ = res.fittedvalues

fig,ax = plt.subplots(figsize = (8,6))

ax.plot(x,y,'o',label = 'data') ## 原始数据
ax.plot(x,y,'r--',label = 'test')  ## 拟合数据
ax.legend(loc='best')
plt.show()
