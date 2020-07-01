# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv(r'D:\data\python\machine\diabetes.csv')
x_test1 = pd.read_csv(r'D:\data\python\machine\diabetes1.csv')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],
                                                    stratify=diabetes['Outcome'], random_state=66)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)
print("Accuracy on training set:{:.3f}".format(rf.score(x_train, y_train)))
print("Accuracy on test set:{:.3f}".format(rf.score(x_test, y_test)))


# 没有更改任何参数的随机森林有78.6%的准确度，比逻辑回归和单一决策树的预测效果更好。
# 再试试调整max_features设置，看看效果是否能够提高。
# 可以看到结果并没有提高，这表明默认参数的随机森林在这里效果很好。
rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(x_train, y_train)
print("Accuracy on training set:{:.3f}".format(rf1.score(x_train, y_train)))
print("Accuracy on test set:{:.3f}".format(rf1.score(x_test, y_test)))

print('---------------------------------------------- x_train-------------------------------------------------------')
print(x_train)
### 预测用户
x_test1['score'] = rf1.predict(x_test1)

#将DataFrame存储为csv,index表示是否显示行名，default=True

print('-------------------------------------------------')
x_test1.to_csv('D:/data/python/machine/sjsl_test.csv')


# 用可视化的方式来看一下用三种不同正则化参数C所得模型的系数。
# 更强的正则化(C = 0.001)会使系数越来越接近于零。仔细地看图，
# 我们还能发现特征“DiabetesPedigreeFunction”（糖尿病遗传函数）在 C=100, C=1 和C=0.001的情况下, 系数都为正。
# 这表明无论是哪个模型，DiabetesPedigreeFunction（糖尿病遗传函数）这个特征值都与样本为糖尿病是正相关的。
diabetes_features = [x for i, x in enumerate(diabetes.columns) if i != 8]

# 可视化特征重要度，可以从图中看出，血糖是最重要的特征
# diabetes_features=[x for i,x in enumerate(diabetes.columns) if i!=8]
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8, 6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    # plt.barh(range(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


# 随机森林的特征重要度：
# 与单一决策树相似，随机森林的结果仍然显示特征“血糖”的重要度最高，
# 但是它也同样显示“BMI（身体质量指数）”在整体中是第二重要的信息特征。
# 随机森林的随机性促使算法考虑了更多可能的解释，这就导致随机森林捕获的数据比单一树要大得多。
plot_feature_importances_diabetes(rf1)