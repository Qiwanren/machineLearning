import numpy as np
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer

## 创建特征
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

print(feature)

### 创建one-hot编码器
one_hot = LabelBinarizer()

### 对特征进行one-hot编码
dumpY = one_hot.fit_transform(feature)

print(dumpY)