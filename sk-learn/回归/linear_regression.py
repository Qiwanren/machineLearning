#线性回归算法
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#模拟数据

#通过指定开始值、终值和步长创建一维等差数组，但其数组中不包含终值
x = np.linspace(0,10,50)

#uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
#size是生成随机数的数量
#uniform模块不能直接访问，必须导入random模块
noise = np.random.uniform(-2,2,size=50)
y = 5 * x + 6 + noise

#创建模型
liner = LinearRegression()

#拟合模型
#如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值
#numpy.reshape(a, newshape, order=‘C’)[source]
#50行一列（50*1）
liner.fit(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
print(liner)

#预测
y_pred = liner.predict(np.reshape(x,(-1,1)))
#figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
# num:图像编号或名称，数字为编号 ，字符串为名称
# figsize:指定figure的宽和高，单位为英寸；
# dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80      1英寸等于2.5cm,A4纸是 21*30cm的纸张
# facecolor:背景颜色
# edgecolor:边框颜色
# frameon:是否显示边框
plt.figure(figsize=(5,5))
plt.scatter(x,y)
plt.plot(x,y_pred,color='r')
plt.show()
#corf_斜率
#intercept_截距
print(liner.coef_)
print(liner.intercept_)