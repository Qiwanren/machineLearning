#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.datasets.base import Bunch   #引入Bunch类
import _pickle as pickle

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer    # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer     # TF-IDF向量生成类

# 读取Bunch对象
def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch

def writebunchobj(path,bunchobj):
    file_obj = open(path,"wb")
    pickle.dump(bunchobj,file_obj)
    file_obj.close()

if __name__ == "__main__":
    path = "D:/data/machineLearn/pyspark/200222/200222.txt"
    bunch = readbunchobj(path)

    # 构建TF-IDF词向量空间对象
    tfidfspace = Bunch(target_name = bunch.target_name,label = bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})


