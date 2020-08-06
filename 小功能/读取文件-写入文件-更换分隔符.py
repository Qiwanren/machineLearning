'''
    将文件中的分隔符由空格替换为逗号
'''

import re

f = open("D:\data\python\work\woyinyue_order_5gcailing_user_all.txt", "r",encoding='utf-8')   #获取文件对象
file=open('D:\data\python\work\woyinyue_order_5gcailing_user_all_11.txt','w',encoding='utf-8')

line = f.readline()
line = line[:-1]
while line:             #直到读取完文件
    line = f.readline()  #读取一行文件，包括换行符
    line = line[:-1]     #去掉换行符，也可以不去
    list = line.split(' ')
    count = 0
    for str1 in list:
        if str1 != '':

            ## 去除特殊字符
            str2 = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", str1)
            strs = strs + str2 + "|"
            count +=1
    file.write(strs[:-1]);
    file.write('\n')
f.close() #关闭文件
file.close()