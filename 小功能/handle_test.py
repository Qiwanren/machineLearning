# code=utf-8
import re

str = "['99', '100', '720', '20']"

list1 = str.split(',')
str1 = re.sub("\D", "", list1[-1])
print(str1)