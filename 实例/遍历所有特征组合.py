import sys
from itertools import combinations

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn import preprocessing

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("D:/wo_yinyue_tezheng1.txt")  # 保存到D盘

lbl = preprocessing.LabelEncoder()
from time import strftime, localtime

# 打印当前时间
def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return



def handleFlagField(x):
    if x !=1 and x !=0:
        return 0
    else:
        return x
'''
    指定数据字段的类型及处理空值
'''
def changeType(data):
    data['prov_id'].fillna('099', inplace=True)
    data['prov_id'] = lbl.fit_transform(data['prov_id'].astype(str))  # 将提示的包含错误数据类型这一列进行转换
    data['user_id'].fillna('9999999999', inplace=True)
    data['user_id'] = lbl.fit_transform(data['user_id'].astype(str))
    data['product_id'].fillna('9999999', inplace=True)
    data['product_id'] = lbl.fit_transform(data['product_id'].astype(str))
    data['area_id'].fillna('0991', inplace=True)
    data['area_id'] = lbl.fit_transform(data['area_id'].astype(str))
    data['device_number'].fillna('9999999999', inplace=True)
    data['device_number'] = lbl.fit_transform(data['device_number'].astype(str))
    data['cust_sex'].fillna('1', inplace=True)
    data['cust_sex'] = lbl.fit_transform(data['cust_sex'].astype(str))
    data['cert_age'].fillna(data['cert_age'].mean(), inplace=True)
    data['cert_age'] = lbl.fit_transform(data['cert_age'].astype(int))
    data['total_fee'].fillna(data['total_fee'].mean(), inplace=True)
    data['total_fee'] = lbl.fit_transform(data['total_fee'].astype(float))
    data['jf_flux'].fillna(data['jf_flux'].mean(), inplace=True)
    data['jf_flux'] = lbl.fit_transform(data['jf_flux'].astype(float))
    data['fj_arpu'].fillna(data['fj_arpu'].mean(), inplace=True)
    data['fj_arpu'] = lbl.fit_transform(data['fj_arpu'].astype(float))
    data['ct_voice_fee'].fillna(data['ct_voice_fee'].mean(), inplace=True)
    data['ct_voice_fee'] = lbl.fit_transform(data['ct_voice_fee'].astype(float))
    data['total_flux'].fillna(data['total_flux'].mean(), inplace=True)
    data['total_flux'] = lbl.fit_transform(data['total_flux'].astype(float))
    data['total_times'].fillna(data['total_times'].mean(), inplace=True)
    data['total_times'] = lbl.fit_transform(data['total_times'].astype(int))
    data['total_nums'].fillna(data['total_nums'].mean(), inplace=True)
    data['total_nums'] = lbl.fit_transform(data['total_nums'].astype(int))
    data['in_cnt'].fillna(data['in_cnt'].mean(), inplace=True)
    data['in_cnt'] = lbl.fit_transform(data['in_cnt'].astype(int))
    data['out_cnt'].fillna(data['out_cnt'].mean(), inplace=True)
    data['out_cnt'] = lbl.fit_transform(data['out_cnt'].astype(int))
    data['in_dura'].fillna(data['in_dura'].mean(), inplace=True)
    data['in_dura'] = lbl.fit_transform(data['in_dura'].astype(int))
    data['out_dura'].fillna(data['out_dura'].mean(), inplace=True)
    data['out_dura'] = lbl.fit_transform(data['out_dura'].astype(int))
    data['lianxi_user'].fillna(data['lianxi_user'].mean(), inplace=True)
    data['lianxi_user'] = lbl.fit_transform(data['lianxi_user'].astype(int))
    data['out_dura'].fillna(data['out_dura'].mean(), inplace=True)
    data['out_dura'] = lbl.fit_transform(data['out_dura'].astype(str))
    data['brand'].fillna('其他', inplace=True)
    data['brand'] = lbl.fit_transform(data['brand'].astype(str))
    data['one_city_flag'].fillna('0', inplace=True)
    data['one_city_flag'] = lbl.fit_transform(data['one_city_flag'].astype(str))
    data['flag'].fillna('0', inplace=True)
    data['flag'] = data['flag'].apply(lambda x: handleFlagField(x))
    return data


printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result4.txt'

params = ['prov_id', 'user_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age', 'total_fee',
          'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_times', 'total_nums', 'in_cnt', 'out_cnt',
          'in_dura', 'out_dura', 'lianxi_user', 'one_city_flag', 'brand','app_visit_cnt','app_visit_dura','app_up_flow','app_down_flow','app_total_flow','app_active_days', 'flag']

train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=params, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep="|", names=params, encoding='utf-8')

train = changeType(train)
test = changeType(test)

## 生成DMatrix,字段内容必须为数字或者boolean
x_featur_params1 = ['cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux',
                   'total_times', 'total_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'lianxi_user', 'one_city_flag','brand','app_visit_cnt','app_visit_dura','app_up_flow','app_down_flow','app_total_flow','app_active_days']

result_list = sum([list(map(list, combinations(x_featur_params1, i))) for i in range(len(x_featur_params1) + 1)], [])
x_featur_params = []
for arr in result_list:
    if len(arr) > 9:
        x_featur_params = arr
        print("x_featur_params = " + str(x_featur_params))
        xgb_train = xgb.DMatrix(train[x_featur_params],train['flag'])
        xgb_test = xgb.DMatrix(test[x_featur_params])
        y_test = test['flag']

        ## 定义模型训练参数
        params = {
        'booster': 'gbtree',   ####  gbtree
        'objective': 'binary:logistic',  # 多分类的问题  'objective': 'binary:logistic' 二分类，multi:softmax 多分类问题
        'gamma': 0.01,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 10,               # 构建树的深度，越大越容易过拟合
        'min_child_weight': 2,
        'eta': 0.01,                  # 如同学习率
        'learning_rate':0.1,
        'subsample':0.5,
        'colsample_bytree':1.0,
        'reg_alpha':1.0
        }

        ## 寻找最佳的 n_estimators


        plst = params.items()
        ## 训练轮数
        num_round = 10

        ## 模型训练
        model = xgb.train(params,xgb_train,num_round)

        ## 分析特征值
        importance = model.get_fscore()
        importance = sorted(importance.items(),key=lambda x : x[1],reverse=True)

        ans = model.predict(xgb_test)
        test['score'] = ans
        print('预测值AUC为 ：%f'% roc_auc_score(y_test,ans))


        # 将结果输出到文件
        #test.to_csv('D:/data/python/work/result.csv')

        # 显示重要特征
        #plot_importance(model)
        #plt.show()

        printTime()