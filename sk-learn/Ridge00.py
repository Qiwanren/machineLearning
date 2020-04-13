from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mglearn

'''
    岭回归泛化能力优于线性回归，带来的就是训练集精度下降，测试集精度上升。
    该模型支持alpha参数，该参数默认为1，调大alpha会进一步下降训练集精度，可能加强泛化能力；相反，调小alpha则减少了约束，训练集精度上升，可能降低泛化能力。
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

# 波士顿extended数据集
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
