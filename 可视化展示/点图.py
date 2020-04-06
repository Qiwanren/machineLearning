import matplotlib.pyplot as plt

# 绘制散点图(传如一对x和y坐标，在指定位置绘制一个点)
'''
    plt.scatter(2, 4)
    # 设置输出样式
    plt.scatter(3, 5, s=200)
'''
import numpy as np

np.random.seed(1)
x = np.random.rand(10)
y = np.random.rand(10)

x1 = np.random.rand(10)
y1 = np.random.rand(10)

colors = np.random.rand(10)
area = (30 * np.random.rand(10)) ** 2
plt.scatter(x, y, s=200, c='b', alpha=0.5, )  ####  蓝色点图
plt.scatter(x1, y1, s=200, c='r', alpha=0.5, )  ### 红色点图
plt.show()