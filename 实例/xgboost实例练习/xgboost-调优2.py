import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

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
x_featur_params = ['cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux',
                   'total_times', 'total_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'lianxi_user', 'one_city_flag','brand','app_visit_cnt','app_visit_dura','app_up_flow','app_down_flow','app_total_flow','app_active_days']

label = ['flag']

x_train= train[x_featur_params]
y_train = train['flag']
x_test = test[x_featur_params]
y_test = test['flag']


## 初始化参数
other_params = {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 100,'min_child_weight':2,'gamma':0.01,
                'objective':'binary:logistic','booster':'gbtree','subsample':0.5,'colsample_bytree':1.0,'reg_alpha':1.0}

## 寻找最佳的 n_estimators
cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}

## 寻找最佳的 max_depth
cv_params = {'max_depth': np.linspace(1, 20, 20, dtype=int)}

#寻找最佳的 min_child_weight
cv_params = {'min_child_weight': np.linspace(1, 10, 10, dtype=int)}

# 寻找最佳的gamma值
cv_params = {'gamma': np.linspace(0, 0.1, 11)}

# 寻找最佳的subsample值
cv_params = {'subsample': np.linspace(0, 1, 11)}

#寻找最佳的colsample_bytree值
cv_params = {'colsample_bytree': np.linspace(0, 1, 11)[1:]}

# 寻找最佳的reg_lambda
cv_params = {'reg_lambda': np.linspace(0, 100, 11)}

# 输出最优的reg_alpha为默认值，继续细分
cv_params = {'reg_alpha': np.linspace(0, 1, 11)}

# 寻找最佳的 eta 值
cv_params = {'eta': np.logspace(-2, 0, 10)}

regress_model = xgb.XGBRegressor(**other_params)  # 注意这里的两个 * 号！
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
gs.fit(x_train, y_train)  # X为训练数据的特征值，y为训练数据的label

# 性能测评
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)


