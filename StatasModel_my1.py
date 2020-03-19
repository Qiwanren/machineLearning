#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

### Y = 5+2.X + 3.X²
nsample = 50
x = np.linspace(0,10,nsample)    ###  从零到十之间，选取nsample个数
X = np.column_stack((x,x**2))
X = sm.add_constant(X)

### 设置 β0 和  β1 的值分别为2,5
beta = np.array([5,2,3])

e = np.random.normal(size=nsample)
y = np.dot(X,beta) + e      ###构造函数

model = sm.OLS(y,X)
results = model.fit()
results.params
results.summary()

y_fitted = results.fittedvalues
fig,ax = plt.subplots(figsize = (8,6))
ax.plot(x,y,'o',label = 'data') ## 原始数据
ax.plot(x,y_fitted,'r--',label = 'OLS')  ## 拟合数据
ax.legend(loc='best')
plt.show()
