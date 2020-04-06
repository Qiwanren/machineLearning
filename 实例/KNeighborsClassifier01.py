from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
cancer = load_breast_cancer()

# 切分
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# 记录不同n_neighbors情况下，模型的训练集精度与测试集精度的变化
training_accuracy = []
test_accuracy = []

# n_neighbors取值从1到10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # 模型对象
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    # 训练
    clf.fit(X_train, y_train)
    # 记录训练集精度
    training_accuracy.append(clf.score(X_train, y_train))
    # 记录测试集精度
    test_accuracy.append(clf.score(X_test, y_test))

# 画出2条曲线，横坐标是邻居个数，纵坐标分别是训练集精度和测试集精度
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()