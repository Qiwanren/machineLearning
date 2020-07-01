# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.svm import SVC

diabetes = pd.read_csv(r'D:\data\python\machine\diabetes.csv')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],
                                                    stratify=diabetes['Outcome'], random_state=66)



# SVM要求所有的特征要在相似的度量范围内变化。我们需要重新调整各特征值尺度使其基本上在同一量表上。
# 从这个结果中，我们可以看到数据的度量标准化后效果大不同！现在我们的模型在训练集和测试集的结果非常相似，
# 这其实是有一点过低拟合的，但总体而言还是更接近100%准确度的。
# 这样来看，我们还可以试着提高C值或者gamma值来配适更复杂的模型。
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

svc = SVC()
svc.fit(x_train_scaled, y_train)
print("Accuracy on training set:{:.2f}".format(svc.score(x_train_scaled, y_train)))
print("Accuracy on test set:{:.2f}".format(svc.score(x_test_scaled, y_test)))