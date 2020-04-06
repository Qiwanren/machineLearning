import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

iris_datas = load_iris()

target = iris_datas.target;
data = iris_datas.data;

# 拆分数据，获取测试集和训练集
'''
X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.4, random_state=0,stratify=y_train)
    # train_data：所要划分的样本特征集
    # train_target：所要划分的样本结果
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。
        # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
    stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下： 
        training: 75个数据，其中60个属于A类，15个属于B类。 
        testing: 25个数据，其中20个属于A类，5个属于B类。 
        用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。
        将stratify=X就是按照X中的比例分配 
        将stratify=y就是按照y中的比例分配 
    返回值介绍：
        x_train,y_train 为训练集的样本集和结果集
        x_test,y_test   为测试集的样本集和结果集
'''
x_train,x_test,y_train,y_test = train_test_split(data,target,random_state=0)

'''
# 观察数据
## 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(x_train,columns=iris_datas.feature_names)

# 利用Dataframe创建散点图矩阵，按y_train着色
grr = pd.pandas.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',
                        hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
plt.show()

'''

'''
    原理:K近邻算法，即是给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的K个实例，这K个实例的
多数属于某个类，就把该输入实例分类到这个类中。

'''

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

## 创建一个测试值，然后进行预测
x_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(x_new)
## 输出结果
print(prediction)

## 评估模型
y_pred = knn.predict(x_test)
## 获取准确率
yes_rate = np.mean(y_test==y_pred)
yes_rate1 = knn.score(x_test,y_test)
print(yes_rate)
print(yes_rate1)




