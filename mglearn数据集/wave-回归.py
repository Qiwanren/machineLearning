import matplotlib.pyplot as plt
import mglearn
# 构造40个样本


X, y = mglearn.datasets.make_wave(n_samples=40)
print(X)
print('--------------------------------------------------')
print(y)
# 因为X只有1维, 所以直接可以画散点图
plt.plot(X, y, 'v')
# y的连续值范围
plt.ylim(-3, 3)
# 画横坐标说明
plt.xlabel("Feature")
# 画纵坐标说明
plt.ylabel("Target")
plt.show()