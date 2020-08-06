# code=utf-8
import re
import numpy as np

'''
str = "['99', '100', '720', '20']"
list1 = str.split(',')
str1 = re.sub("\D", "", list1[-1])
print(str1)


str = "51      A5119030658211607       90065148.0      V0445201        8E652DA5A42013DF        1       36      73      0       15      0       11365   253     210 56       154     32      88      8       欧珀    OPPOR11T        0       1"
list = str.split(' ')
strs = ''
for str1 in list:
    if str1 != '':
        strs = strs + str1 +","
print(strs[:-1])

params = ['prov_id', 'user_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age', 'total_fee',
          'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_times', 'total_nums', 'in_cnt', 'out_cnt',
          'in_dura', 'out_dura', 'lianxi_user', 'brand', 'brand_detail', 'one_city_flag', 'flag']

print(params.values)

'''
def method0():
    ans = [1,2,3,4,5,6,7]
    ans_len = len(ans)
    for row in range(0, ans_len):
        print(ans[row])


def method1():
    str1 = "51   A5119123052749904    90063345.0   V0440600    000AF6FD12B21172    2    27   49   0    30   0    36561  15   11 NULL   NULL  1    1"
    list = str1.split(" ")
    print(len(list))
    for str in list:
        str1 = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", str)
        if str1 != '':
            print(str)

def method2():
    #w = np.random.rand(15, 1)
    data = np.random.rand(5, 10)
    print(data)

def method3():
    num_params = ['cust_sex', 'cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_dura',
                  'roam_dura', 'total_times',
                  'total_nums', 'local_nums', 'roam_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'visit_cnt',
                  'visit_dura', 'up_flow', 'down_flow',
                  'total_flow', 'active_days', 'imei_duration', 'avg_duratioin']

    n = 4.56
    for com in num_params:
        print(com + ' : ',n)

if __name__ == '__main__':
    method3()