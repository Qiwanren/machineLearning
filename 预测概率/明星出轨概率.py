# -*- coding: UTF-8 -*-
import findspark
findspark.init()
from pyspark.sql import SparkSession

spark=SparkSession \
        .builder \
        .appName('predict rate') \
        .getOrCreate()

