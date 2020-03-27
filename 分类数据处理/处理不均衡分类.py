import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

### 加载莺尾花数据
iris = load_iris()  ### 返回一个字典类型数据

### 创建特征矩阵
features = iris.data

### 创建目标向量
target = iris.target

# 移除个观察值
features = features[40:,:]
#print(target)
target = target[40:]

#创建二元目标向量来标识观察值是否为类别 0
target = np.where((target == 0),0,1)   ###
#print(target)

### 给每个分类的观察值打标签
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

print(iris)
print(i_class0)
print(i_class1)
print('----------------------------------------------')

# 确定每个分类的观察值数量
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# 对于每个分类为 0  的观察值，从分类为 1 的数据中进行无放回的随机采样
i_class1_downsampled = np.random.choice(i_class1,size=n_class0,replace=False)

# 将分类为 0 的目标向量和下采样的分类为 1 的目标向量连接起来
arr1 = np.hstack((target[i_class0],target[i_class1_downsampled]))

# 将分类为 0 的特征矩阵和下采样的分类为 1 的特征矩阵连接起来
arr2 = np.hstack((features[i_class0,:],features[i_class1_downsampled,:]))[0:5]
#print(arr2)

# 对于每个分类为 1 的观察值，从分类为 0 的数据中进行有放回的随机采样
i_class0_upsampled = np.random.choice(i_class0,size=n_class1,replace=True)
print(i_class0_upsampled)

# 将上采样得到的分类为0的目标向量和分类为1的目标向量连接起来
arr3 =  np.concatenate((target[i_class0_upsampled],target[i_class1]))
print(arr3)
