# !/usr/bin/python
# -*- coding:utf-8 -*-
# File Name: connect_utils.py
# Author: Zihan Qin
# Mail: zihanqin@usc.edu
# Created Time: 2021-11-27 00:13:07


import re
import urllib
from collections import namedtuple


Features = namedtuple("features",
    "id image_id height width nChannels mode sift color_moments color_gist")


class SparkRDS:
    def __init__(self, spark):
        self.spark = spark
        self.rds_prop = {
            'user': 'admin',
            'password': 'wikiartdb',
            'driver': 'com.mysql.cj.jdbc.Driver'
        }
        self.rds_url = 'jdbc:mysql://wikiartdb.cv8worynfzsx.us-west-1.rds.amazonaws.com:3306/wikiart'

    @property
    def last_feature_id(self):
        data = self.spark.read.jdbc(url=self.rds_url, table='photos_features', properties=self.rds_prop)
        return max([each.id for each in data.select(data.id).collect()])

    def upload_features(self, features):
        print("begin to uploading RDS...")
        df = self.spark.createDataFrame(features, schema=list(Features._fields))
        df.write.jdbc(url=self.rds_url, table='photos_features', mode='append', properties=self.rds_prop)

    def download_features(self, name, return_type="dict"):
        data = self.spark.read.jdbc(url=self.rds_url, table='photos_features', properties=self.rds_prop)
        data = data.filter(data.image_id == name).collect()[0]
        if return_type == "Row":
            return data
        elif return_type == "dict":
            data = data.asDict()
        else:
            raise NotImplementedError("%s is not implemented" % return_type)
        return data

    def get_path(self, name):
        print("getting path of %s from RDS... " % name)
        data = self.spark.read.jdbc(url=self.rds_url, table='photos_picture', properties=self.rds_prop)
        try:
            path = data.filter(data.name == name).select(data.url_path).collect()[0]
        except IndexError as e:
            print("figure %s not in S3" % name)
        path = path.asDict()["url_path"]
        return path


class SparkS3:
    def __init__(self,
        spark,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY
    ):
        self.spark = spark
        self.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
        self.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY

    def download_image(self, http_path):
        print("downloading %s from S3..." % http_path)
        s3a_path = self.http_to_s3a(http_path)
        try:
            fig = self.spark.read.format('image').option("dropInvalid", True).load(s3a_path)
        except Exception as e:
            print("s3a_path %s is not in S3" % s3a_path)
        return fig.collect()[0].image

    def http_to_s3a(self, path):
        path = path.replace("'", "").replace("https", "s3a")
        path = path.replace("acrawdata.s3.amazonaws.com", "acrawdata")
        path = re.sub(r'\?.*$', '', path)
        path = urllib.parse.unquote(path)
        return path
