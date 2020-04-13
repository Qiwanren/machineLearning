from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn

# 生成60个样本数据, 一维特征
X, y = mglearn.datasets.make_wave(n_samples=60)
# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# 训练线性回归模型
lr = LinearRegression().fit(X_train, y_train)

# coef_就是斜率w, 即每个特征对应一个权重
print("lr.coef_: {}".format(lr.coef_))
# intercept_是截距b
print("lr.intercept_: {}".format(lr.intercept_))

# 训练集精度
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# 测试机精度
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))