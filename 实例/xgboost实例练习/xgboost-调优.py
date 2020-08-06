import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from matplotlib import pyplot as plt
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

x_train= train[x_featur_params]
y_train = train['flag']
x_test = test[x_featur_params]
y_test = test['flag']

## 测试不同的深度
arrs = np.random.randint(5,20,size=15)

for arr in arrs:
    ## 模型训练
    model = xgb.XGBClassifier(max_depth=arr, learning_rate=0.1, n_estimators=50, silent=True, objective='binary:logistic',
                              booster='gbtree')
    model.fit(x_train, y_train)

    # 对测试集进行预测，并计算AUC
    ans = model.predict(x_test)
    test['score'] = ans
    print('预测值AUC为 ：%f' % roc_auc_score(y_test, ans))

    for arr in arrs:
        ## 模型训练
        model = xgb.XGBClassifier(max_depth=arr,learning_rate=0.1,n_estimators=50,silent=True,objective='binary:logistic',
                                                booster='gbtree')
        model.fit(x_train,y_train)

        # 对测试集进行预测，并计算AUC
        ans = model.predict(x_test)
        test['score'] = ans
        print('预测值AUC为 ：%f'% roc_auc_score(y_test,ans))

'''
## 获得特征重要性评分
importances = model.feature_importances_

# 对特征重要性去重作为候选阈值
thresholds = []
for importance in importances:
    if importance not in thresholds:
        thresholds.append(importance)

## 候选阈值排序
thresholds = sorted(thresholds)


## 遍历候选阈值
for threshold in thresholds:
    ## 通过threshold 进行特征选择
    selection = SelectFromModel(model,threshold = threshold,prefit=True)
    select_X_train = selection.transform(x_train)

    # 训练模型
    selection_model = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=50,silent=True,objective='binary:logistic',
                                        booster='gbtree')
    selection_model.fit(select_X_train,y_train)
    ## 评估模型
    select_X_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_X_test)
    auc = roc_auc_score(y_test,y_pred)
    print("阈值：%.3f,  特征数量：%d,  AUC得分: %.2f" %(threshold,select_X_train.shape[1],auc))

'''
# 将结果输出到文件
#test.to_csv('D:/data/python/work/result.csv')

# 显示重要特征
#plot_importance(model)
#plt.show()