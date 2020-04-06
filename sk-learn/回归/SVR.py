import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

import sklearn.datasets as datasets

#支持向量回归
from sklearn.svm import SVR

#2.生成训练数据
x = np.linspace(-np.pi,np.pi,60)
y = np.sin(x)

#数据加噪

y[::3]+=0.5-np.random.random(20)
X_train = x.reshape(-1,1)
Y_train = y

#3.创建支持向量回归模型
svr_linear = SVR(kernel='linear')
svr_rbf = SVR(kernel='rbf')
svr_poly = SVR(kernel='poly')

#4.训练数据

svr_linear.fit(X_train,Y_train)
svr_rbf.fit(X_train,Y_train)
svr_poly.fit(X_train,Y_train)

#5.与测试数据

#获取预测数据自变量范围
xmin,xmax = X_train.min(),X_train.max()
x_test = np.arange(xmin,xmax,0.01).reshape(-1,1)
#获取预测数据
linear_y_pre = svr_linear.predict(x_test)
rbf_y_pre = svr_rbf.predict(x_test)
poly_y_pre = svr_poly.predict(x_test)

#6.绘图
results = [linear_y_pre,rbf_y_pre,poly_y_pre]
titles = ['Linear','rbf','poly']
plt.figure(figsize=(12,12))
for i,result in enumerate(results):
    plt.subplot(3,1,i+1)
    plt.scatter(X_train,Y_train)
    plt.plot(x_test,result,color='orange')
    plt.title(titles[i])