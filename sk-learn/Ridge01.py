import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归

'''
    Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
        该模型支持alpha参数，该参数默认为1，调大alpha会进一步下降训练集精度，可能加强泛化能力；
        相反，调小alpha则减少了约束，训练集精度上升，可能降低泛化能
    RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
        参数名：alphas
            类型： numpy array of shape [n_alphas]
            说明：正则化的力度，必须是正浮点数。正则化提升了问题的条件，减少了估计器的方差。较大的值指定了更强的正则化
        参数名：fit_intercept
            类型：boolean
            说明：是否计算该模型的截距。如果设置为False，将不会在计算中使用截距（比如，预处理数据已经中心化）
        参数名：normalize
            类型：boolean, optional, default False
            说明：当fit_intercept设置为False时，该参数将会被忽略。如果为True，则回归前，回归变量X将会进行归一化，减去均值，然后除以L2范数。如果想要标准化，请在评估器（normalize参数为False）调用fit方法前调用sklean.preprocessing.StandardScaler
        参数名：scoring
            类型：string, callable or None, optional, default: None
            说明：一个字符串(见模型评估文档)或一个评估分数的可调用的对象/函数，对象/函数带有注册的评估器(estimator, X, y)
        参数名：gcv_mode
            类型： {None, ‘auto’, ‘svd’, eigen’}, optional
            说明：说明在执行通用交叉验证时使用的策略的标志。选项有:          
            auto:如果n_samples为> n_feature，或者当X为稀疏矩阵时，使用svd，否则使用eigen。
            svd:用X奇异值分解的力计算(不适用于稀疏矩阵)
            eigen:力的计算通过eigen分解 XTXXTX
            “auto”模式是默认的，它的目的是根据训练数据的形状和格式选择两个更廉价的选项
        属性
            参数名：cv_values_
                类型：array, shape = [n_samples, n_alphas] or shape = [n_samples, n_targets, n_alphas], optional
                说明：每个alpha的交叉验证值(如果store_cv_values=True和cv=None)。在fit()被调用之后，这个属性将包含均方差(默认值)或{loss,score}_func函数(如果在构造函数中提供)。
            参数名：coef_
                类型： array, shape = [n_features] or [n_targets, n_features]
                说明：权重向量
            参数名：intercept_
                类型：float | array, shape = (n_targets,)
                说明：决策函数中的独立项，即截距，如果fit_intercept=False,则设置为0
            参数名：alpha_
                类型： float
                说明：正则化参数估计。

'''
# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
data=[
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
]

#生成X和y矩阵
dataMat = np.array(data)
X = dataMat[:,0:1]   # 变量x
y = dataMat[:,1]   #变量y

# ========岭回归========
model = Ridge(alpha=0.5)
model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X, y)   # 线性回归建模
print('系数矩阵:\n',model.coef_)
print('线性回归模型:\n',model)
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
# 使用模型预测
predicted = model.predict(X)

# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='x')
plt.plot(X, predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()