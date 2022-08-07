# !/usr/bin/python
# -*- coding:utf-8 -*-
# File Name: load_feature.py
# Author: Zihan Qin
# Mail: zihanqin@usc.edu # Created Time: 2021-10-27 15:45:41


import os
import re
import cv2
import image_process
import numpy as np
import json
import urllib
from collections import namedtuple
from tqdm import tqdm
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from multiprocessing import Pool

AWS_ACCESS_KEY_ID = "AKIA3UZI2QOSQSCCDTM4"
AWS_SECRET_ACCESS_KEY = "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"

spark = SparkSession.builder\
            .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)\
            .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)\
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
            .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

rds_prop = {
    'user': 'admin',
    'password': 'wikiartdb',
    'driver': 'com.mysql.cj.jdbc.Driver'
}
rds_url = 'jdbc:mysql://wikiartdb.cv8worynfzsx.us-west-1.rds.amazonaws.com:3306/wikiart'

Features = namedtuple("features",
    "http_path s3a_path height width nChannels mode sift color_moments color_gist")


def http_to_s3a(path):
    path = path.replace("'", "").replace("https", "s3a")
    path = path.replace("acrawdata.s3.amazonaws.com", "acrawdata")
    path = re.sub(r'\?.*$', '', path)
    path = urllib.parse.unquote(path)
    return path


def extract_features(path):
    s3a_path = http_to_s3a(path)
    fig = spark.read.format('image').option("dropInvalid", True).load(s3a_path)
    fig = fig.collect()[0].image
    image = np.array(fig.data, dtype=np.uint8).reshape(fig.height, fig.width, fig.nChannels)
    sift_feature = image_process.sift(image)
    color_moments_feature = image_process.color_moments(image)
    color_gist = image_process.color_gist(image)
    features = Features(
        http_path = path,
        s3a_path = fig.origin,
        height = fig.height,
        width = fig.width,
        nChannels = fig.nChannels,
        mode = fig.mode,
        sift = json.dumps(sift_feature.tolist()),
        color_moments = json.dumps(color_moments_feature),
        color_gist = json.dumps(color_gist.tolist())
    )
    return features

print("reading train data information from RDS...")
num_thread = 5
data = spark.read.jdbc(url=rds_url, table='train_data', properties=rds_prop)
paths = data.select(data.x_path).collect()

def transform(paths):
    debug = True
    all_features = []
    for path in tqdm(paths):
        try:
            path = path.asDict()["x_path"]
            features = extract_features(path)
            all_features.append(tuple(features))
        except Exception as e:
            print(path)
    return all_features

print("begin to process features")
all_features = transform(paths)

print("begin to uploading features...")
df = spark.createDataFrame(all_features, schema=list(Features._fields))
df.write.jdbc(url=rds_url, table='features', mode='overwrite', properties=rds_prop)
