#!/usr/bin/env python
# -*- coding:utf-8 -*-
import findspark
findspark.init()

from operator import add

from pyspark import SparkContext


if __name__ == "__main__":
    sc = SparkContext(appName="PythonWordCount")
    ## 读取文件
    lines = sc.textFile('D:/file/hadoop/input/word.txt')
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))
    sc.stop()