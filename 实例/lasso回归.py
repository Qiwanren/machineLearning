from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import mglearn

'''
    与岭回归类似，采用了另外一种正则化叫做L1正则化，它可以约束某些w系数为0，相当于自动筛掉了一些没用的特征

'''
# 波士顿extended数据集
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
# lasso.coef_是w斜率向量，数一下有几个特征的系数不为0
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))