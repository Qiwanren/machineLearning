import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#创建一个总体(随机选取500个从1到500的服从均匀分布的数值)
population = np.random.randint(1,501,500)

#定义函数。该函数定义为从总体data中不放回的抽取样本100次，每次抽取n个数值。并返回这100个样本的平均无偏方差和平均有偏方差。
def funct(data,n):
    a = []
    b = []
    for i in range(100):
        sample_data = np.random.choice(a=data,size=n,replace=False) #replace=False表示不放回取样
        #numpy.var()计算方差时除以的是n，pandas.var()计算方差时除以的是n-1.
        s1 = sample_data.var()   #有偏样本方差
        s2 = pd.Series(sample_data).var()   #无偏样本方差
        a.append(s1)
        b.append(s2)
    bias_var = sum(a)/len(a)    #100个样本的平均有偏样本方差
    nobias_var = sum(b)/len(b)  #100个样本的平均无偏样本方差
    return bias_var,nobias_var

#对样本量n分别从1取到500，重复上面的操作
x = np.linspace(10,500,500)
y1 = []
y2 = []
for num in range(1,501):
    y1.append(funct(population,num)[0])
    y2.append(funct(population,num)[1])

#绘图
plt.rcParams['font.sans-serif']=['SimHei']#黑体
plt.figure(dpi=400)

plt.scatter(x,y1,s=2,alpha=0.6,label="bias_var")

plt.scatter(x,y2,s=2,alpha=0.6,label="nobias_var")

plt.axhline(y=population.var(),c="c",ls="-.",lw=2)

plt.legend(shadow=True)
plt.grid(alpha=0.5)
plt.xlim(0,500)
plt.ylim(15000,23000)
plt.xlabel("样本数量(n)")
plt.ylabel("总体和样本方差")
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.title("样本方差随着样本增大的变化趋势")
plt.savefig("hah.jpg")
plt.show()