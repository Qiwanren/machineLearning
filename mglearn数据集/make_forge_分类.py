import mglearn
import matplotlib.pyplot as plt

# 生成forge样本的特征X和目标y
X, y = mglearn.datasets.make_forge()
print(X)
print('--------------------------------------------------')
print(y)
# 使用样本的第0列特征和第1列特征作为绘制的横坐标和纵坐标，目标y作为图案
#mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
mglearn.discrete_scatter(X[:, 0], X[:, 1],y)
# 在右下角画一个图案的文字说明，即2个分类
plt.legend(["Class 0", "Class 1"], loc=4)
# 绘制横坐标的说明
plt.xlabel("First feature")
# 绘制纵坐标的说明
plt.ylabel("Second feature")
# 样本的个数和特征的维度
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
plt.show()
