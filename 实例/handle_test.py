import sys

import numpy as np
import pandas as pd


def method01():
    arrs = np.random.randint(5, 20, size=15)

    for arr in arrs:
        print(arr)

def method02():
    arrs = range(3, 20, 2)
    for arr in arrs:
        print(arr)

def method03():
    from itertools import combinations
    L = [1, 2, 3, 1]
    result_list = sum([list(map(list, combinations(L, i))) for i in range(len(L) + 1)], [])
    for arr in result_list:
        if len(arr) > 0:
            print(len(arr))
            print("arr" + str(arr))


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def method04():
    sys.stdout = Logger("D:/wo_yinyue_tezheng.txt")  # 保存到D盘
    print('asfasfsaf')
    print('asfasfsaf')

def moded05():
    x = " '96'"
    x1 = x.replace("'","")
    print(x1)

def modeth06(x):
    str = type(x)
    print(str)

if __name__ == '__main__':
    modeth06()