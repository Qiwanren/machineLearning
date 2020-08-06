import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

canceData = load_breast_cancer()
X = canceData.data
y = canceData.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_leaves': 30,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

### 训练集
data_train = lgb.Dataset(X_train, y_train)

## 参数调优
#cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
#                    early_stopping_rounds=50, seed=0)
#print('best n_estimators:', len(cv_results['auc-mean']))
#print('best cv score:', pd.Series(cv_results['auc-mean']).max())

## 实际验证
model=lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metrics='auc',
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=4,
    num_leaves=10,
    max_bin=255,
    min_data_in_leaf=81,
    bagging_fraction=0.7,
    bagging_freq= 30,
    feature_fraction= 0.8,
    lambda_l1=0.1,
    lambda_l2=0,
    min_split_gain=0.1)
model.fit(X_train,y_train)
y_pre=model.predict(X_test)

print(y_pre)

print("acc:",metrics.accuracy_score(y_test,y_pre))
print("auc:",metrics.roc_auc_score(y_test,y_pre))