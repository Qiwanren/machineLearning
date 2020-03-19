#!/usr/bin/env python
# -*- coding:utf-8 -*-
from imp import reload

import jieba
import sys

reload(sys)

'''
    读取文件，对文件内容进行切分，然后将切分结果写入目标文件
'''
def handleText(path1,path2):
    fp1 = open(path1,'r')
    fp2 = open(path2, 'w',encoding="utf-8")
    while True:
        line = fp1.readline()
        # 切分每行的内容
        jieba_list = jieba.cut(line.strip(), cut_all=False)
        str_list = " ".join(jieba_list)
        fp2.write(str_list)
        fp2.write("\n")
        if not line:
            break
    fp1.close()
    fp2.close()


if __name__ == '__main__':
    handleText("D:/data/C3-Art/C3-Art0002.txt","D:/data/machineLearn/out/jieba_deom.txt")