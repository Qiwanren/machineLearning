import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt

lbl = preprocessing.LabelEncoder()

import os
os.environ["PATH"] += os.pathsep + 'D:/anzhuang/graphviz-2.38/bin'


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

all_params = ['prov_id', 'user_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age', 'total_fee',
          'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_times', 'total_nums', 'in_cnt', 'out_cnt',
          'in_dura', 'out_dura', 'lianxi_user', 'one_city_flag', 'brand','app_visit_cnt','app_visit_dura','app_up_flow','app_down_flow','app_total_flow','app_active_days', 'flag']

train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep="|", names=all_params, encoding='utf-8')

print(type(train))

train = changeType(train)
test = changeType(test)


## 生成DMatrix,字段内容必须为数字或者boolean
x_featur_params = ['cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux',
                   'total_times', 'total_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'lianxi_user', 'one_city_flag','brand','app_visit_cnt','app_visit_dura','app_up_flow','app_down_flow','app_total_flow','app_active_days']

label = ['flag']

def ceate_feature_map(features):
    outfile = open('D:/data/python/work/xgb.fmap','w', encoding='utf-8')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat.strip()))
    i = i + 1
    outfile.close()

ceate_feature_map(x_featur_params)

clf = XGBClassifier(
    n_estimators=30,#三十棵树
    learning_rate =0.1,
    max_depth=10,
    min_child_weight=2,
    gamma=0.01,
    subsample=0.5,
    colsample_bytree=1.0,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1
)

clf.fit(train[x_featur_params], train[label])
xgb.to_graphviz(clf)

#plot_tree(clf, num_trees=0, fmap='D:/data/python/work/xgb.fmap')
#fig = plt.gcf()
#fig.set_size_inches(150, 100)
#plt.show()
#fig.savefig('D:/data/python/work/tree.png')