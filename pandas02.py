import pandas as pd

df1 = pd.DataFrame([['Snow', 'M', 22], ['Tyrion', 'M', 32], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                   columns=['name', 'gender', 'age'])

print("----------在最后新增一列---------------")
print("-------案例1----------")
# 在数据框最后加上score一列，元素值分别为：80，98，67，90
df1['score'] = [80, 98, 67, 90]  # 增加列的元素个数要跟原数据列的个数一样
print(df1)

df1.to_csv('D:/data/python/machine/sjsl_test.csv')