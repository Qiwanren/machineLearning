from sklearn import datasets

## 加载手写数字数据集
digits = datasets.load_digits()

## 创建特征矩阵
features = digits.data

## 创建目标向量
target = digits.target

print(target)