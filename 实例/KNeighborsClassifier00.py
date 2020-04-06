from sklearn.model_selection import train_test_split
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
'''
    class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, 
            algorithm=’auto’, leaf_size=30, 
            p=2, metric=’minkowski’, 
            metric_params=None, 
            n_jobs=None, **kwargs)

    参数：
        n_neighbors ： int，optional(default = 5)
            默认情况下kneighbors查询使用的邻居数。就是k-NN的k的值，选取最近的k个点。
        weights ： str或callable，可选(默认=‘uniform’)
            默认是uniform，参数可以是uniform、distance，也可以是用户自己定义的函数。uniform是均等的权重，就说所有
            的邻近点的权重都是相等的。distance是不均等的权重，距离近的点比距离远的点的影响大。用户自定义的函数，接收
            距离的数组，返回一组维数相同的权重。
        algorithm ： {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选
             快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。除此之外，用户也可以自己指定搜索算
             法ball_tree、kd_tree、brute方法进行搜索，brute是蛮力搜索，也就是线性扫描，当训练集很大时，计算非常耗时。kd_tree，
             构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一
             个超矩形，在维数小于20时效率高。ball tree是为了克服kd树高纬失效而发明的，其构造过程是以质心C和半径r分割样本空间，
             每个节点是一个超球体。
         leaf_size ： int，optional(默认值= 30)
            默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的速度和搜索速度，同样也影响着存储树所需的内存大小。
            需要根据问题的性质选择最优的大小。
        p ： 整数，可选(默认= 2)
            距离度量公式。在上小结，我们使用欧氏距离公式进行距离度量。除此之外，还有其他的度量方法，例如曼哈顿距离。这个参数默认为2，
            也就是默认使用欧式距离公式进行距离度量。也可以设置为1，使用曼哈顿距离公式进行距离度量。
         metric ： 字符串或可调用，默认为’minkowski’
            用于距离度量，默认度量是minkowski，也就是p=2的欧氏距离(欧几里德度量)。
        metric_params ： dict，optional(默认=None)
            距离公式的其他关键参数，这个可以不管，使用默认的None即可。
        n_jobs ： int或None，可选(默认=None)
            并行处理设置。默认为1，临近点搜索并行工作数。如果为-1，那么CPU的所有cores都用于并行工作。

'''
X, y = mglearn.datasets.make_forge()
print(X)
print('--------------------------------------------------')
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 初始化模型
# 计算最近3个点中最多出现的分类作为预测标签
clf = KNeighborsClassifier(n_neighbors=3)
## 训练模型
clf.fit(X_train, y_train)
print("Train set accuracy: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

####对数据进行预测,返回样本类别
x_new = np.array([[9.15072323,5.49832246]])
prediction = clf.predict(x_new)
print(prediction)