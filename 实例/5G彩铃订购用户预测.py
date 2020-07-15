import pandas as pd
import numpy as np

## 读取数据,并指定字段名称，分隔符
data=pd.read_csv('D:/data/python/work/5g_cailing_order_user.txt',names=['device_number','product_id','cust_sex','cert_age','total_flux','total_times','total_nums','in_cnt','out_cnt','in_dura','out_dura','total_fee','jf_flux','fj_arpu','ct_voice_fee','city_flag','is_order'],sep=',')

## ,dtype = {'is_order' : str}
print(type(data))
#print(data.info())
### 数据预处理
data['is_order'].fillna('0', inplace=True)
data['product_id'].fillna('9999999', inplace=True)
data['cust_sex'].fillna('1', inplace=True)
data['cert_age'].fillna(data['cert_age'].mean(), inplace=True)
data['total_flux'].fillna(data['total_flux'].mean(), inplace=True)
data['total_times'].fillna(data['total_times'].mean(), inplace=True)
data['total_nums'].fillna(data['total_nums'].mean(), inplace=True)
data['in_cnt'].fillna(data['in_cnt'].mean(), inplace=True)
data['out_cnt'].fillna(data['out_cnt'].mean(), inplace=True)
data['in_dura'].fillna(data['in_dura'].mean(), inplace=True)
data['out_dura'].fillna(data['out_dura'].mean(), inplace=True)
data['total_fee'].fillna(data['total_fee'].mean(), inplace=True)
data['jf_flux'].fillna(data['jf_flux'].mean(), inplace=True)
data['fj_arpu'].fillna(data['fj_arpu'].mean(), inplace=True)
data['ct_voice_fee'].fillna(data['ct_voice_fee'].mean(), inplace=True)
data['city_flag'].fillna('0', inplace=True)

x = data.iloc[1:10,1:5]
print(x)
print(x.corr("spearman"))



#print(data.isnull())  # 是空值返回True，否则返回False
#print(np.any(data.isnull()))  # 只要有一个空值便会返回True，否则返回False


'''
    X = data.iloc[:,3:15]
Y = data.iloc[:,[16]]
print(X.info())
print('------------------------------------------------------------------')
#print(Y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y.astype('int'),test_size = 0.4,stratify=Y, random_state=66)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)
print("Accuracy on training set:{:.3f}".format(rf.score(x_train, y_train)))
print("Accuracy on test set:{:.3f}".format(rf.score(x_test, y_test)))

'''

