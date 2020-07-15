# code=utf-8
file = open('D:/data/python/work/cb_product_info_20200715-222.txt')

file1 = open('D:/data/python/work/cb_product_info_20200715-333.txt','w')    #设置文件对象
#f.write(str)
for line in file.readlines():

     line1=line.split(',')
     if len(line1) >=7:
        str1 = line1[0]
        str2 = line1[1]
        str4 = line1[3]
        str5 = line1[4]
        str6 = line1[5]
        if line1[6].strip() == 'TRUE':
            str3 = line1[2]
        else:
            str3 = ''
        line2 = str1 +','+  str2 +','+ str3 +','+ str4 +','+ str5 +','+ str6
        file1.write(line2)
        file1.write('\n')
