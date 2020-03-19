from IPython.display import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

'''
    贝叶斯看待数据的思维
        假设蹲在山坡上数羊 12,33,20,29,20,30,18（每天经过我面前几只羊，这么一周的数据）
        按照贝叶斯的思想，数据已经定下来了，接下来我要做的就是找到参数的概率分布
        我们的数据是非负的整数，在这里我们用泊松分布建模，泊松分布只需要μ描述数据的均值和方差
        
        p(x|μ) = e^-μ*μ^x for x = 0,1,2.....
        λ = E(x) = Var(μ)
        
        使用泊松分布，观察数据的分布情况
        
        极大似然估计求解μ
        在用贝叶斯之前，先来用一下最大似然估计来求解
          .poisson_logprob()根据泊松模型和参数值返回观测数据的总似然值
          .opt.minmize_scalar找到最合适的取值
          
        
        

'''
plt.style.use('bmh')
colors = ['#348ABD','#A60628','#7A68A6','#467821','#D55E00',
          '#CC79A7','#56B4E9','#009E73','#F0E442','#0072B2']

message = pd.read_csv('D:/data/python/machine/tjfx/hangout_chat_data.csv')

###print(message.head())

fig = plt.figure(figsize=(12,5))
plt.title('Frequency of message by response time')
plt.xlabel('Response time (second)')
plt.ylabel('Number of message')
plt.hist(message['time_delay_seconds'].values,range=[0,60],bins=60,histtype='stepfilled')
#plt.show()

y_obs = message['time_delay_seconds'].values

def poisson_logprob(mu,sign = -1):
    return np.sum(sign.stats.poisson.logpmf(y_obs,mu=mu))

freq_results = opt.minimize_scalar(poisson_logprob)

print(freq_results['x'])