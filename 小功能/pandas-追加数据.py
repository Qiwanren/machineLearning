import datetime
import time




def delete_file_content(f):
    f.seek(0)
    f.truncate()


def write_log(f,content):
    d = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write(d)
    f.write(" - " + content)
    f.write("\n")

'''
def delete_file_content(filename):
    with open(filename, 'r+', encoding='utf-8') as f1:
        f1.seek(0)
        f1.truncate()
    f1.close()
'''



if __name__ == '__main__':
    filename = 'D:/data/python/temp/my_csv.txt'
    f = open(filename, 'a',encoding='UTF-8')
    delete_file_content(f)
    for num in range(10,100):
        write_log(f,"测试内容")
    f.close()
