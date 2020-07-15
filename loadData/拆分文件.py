# code=utf-8
import pandas as pd

# 打开文件
data = pd.read_csv(r'D:/data/python/work/woyinyue-5g-20200708.csv')

# 每个excel保存3万行，那么530000+数据需要18个.csv文档保存
for i in range(0, 31):
    save_data = data.iloc[i*30000 + 1 : (i+1)*30000+1]
    file_name = r'D:/data/python/work/xxx' + str(i) + '.csv'  # 保存文件路径以及文件名称
    save_data.to_csv(file_name, index=False)  # 保存格式为.csv，如果是xlsx则修改为save_data.to_excel