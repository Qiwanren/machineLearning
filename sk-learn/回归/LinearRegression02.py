import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# 加载糖尿病数据集
diabetes = datasets.load_diabetes()

# 只使用一个特征
diabetes_X = diabetes.data[:, np.newaxis, 2]

# 将数据分为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 在测试集上进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)
print('--------------------------------------')
print(diabetes_y_pred)
print('--------------------------------------')

# 系数
print('Coefficients: \n', regr.coef_)
# 均方误差
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# plot绘制 , 绘制原点
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# 绘制直线
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()