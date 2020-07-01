#线性回归算法
import numpy as np

#通过指定开始值、终值和步长创建一维等差数组，但其数组中不包含终值
x = np.linspace(0,10,50)
print(x)
print('--------------------------------------------------------------')
#uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
#size是生成随机数的数量
#uniform模块不能直接访问，必须导入random模块
noise = np.random.uniform(-2,2,size=50)
y = 5 * x + 6 + noise
#print(y)

print('--------------------------------------------------------------')

## argmax方法测试，
a = np.array([3, 1, 2, 4, 6, 1])
#print(np.argmax(a))
