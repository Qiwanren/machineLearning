import sys


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("D:/wo_yinyue_pex_xgxs1111.txt")  # 保存到D盘

## 该文件内容会写入到文件中
print('测试日志功能')