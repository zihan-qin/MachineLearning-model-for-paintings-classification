# !/usr/bin/python
# -*- coding:utf-8 -*-
# File Name: load_feature.py
# Author: Zihan Qin
# Mail: zihanqin@usc.edu # Created Time: 2021-10-27 15:45:41


import image_process
import spark_connection_utils
import numpy as np
import json
from pyspark.sql import SparkSession


class FeatureExtraction:
    def __init__(self,
        spark,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY
    ):
        self.rds_connector = spark_connection_utils.SparkRDS(spark)
        self.s3_connector = spark_connection_utils.SparkS3(spark, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

    def extract_features(self, name):
        print("extracting feature for %s" % name)
        path = self.rds_connector.get_path(name)
        fig = self.s3_connector.download_image(path)
        image = np.array(fig.data, dtype=np.uint8).reshape(fig.height, fig.width, fig.nChannels)
        sift_feature = image_process.sift(image)
        color_moments_feature = image_process.color_moments(image)
        color_gist = image_process.color_gist(image)
        features = spark_connection_utils.Features(
            id = self.last_feature_id+1,
            image_id = name,
            height = fig.height,
            width = fig.width,
            nChannels = fig.nChannels,
            mode = fig.mode,
            sift = json.dumps(sift_feature.tolist()),
            color_moments = json.dumps(color_moments_feature),
            color_gist = json.dumps(color_gist.tolist())
        )
        return features

    @property
    def last_feature_id(self):
        return self.rds_connector.last_feature_id

    def update_feature(self, name):
        features = [self.extract_features(name)]
        self.rds_connector.upload_features(features)


if __name__ == "__main__":
    debug = True
    if debug:
        name = "albrecht-durer_abduction"
    else:
        name = sys.argv[1]
    AWS_ACCESS_KEY_ID = "AKIA3UZI2QOSQSCCDTM4"
    AWS_SECRET_ACCESS_KEY = "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"
    spark = SparkSession.builder\
                .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)\
                .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)\
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
                .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    extract_process = FeatureExtraction(spark, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    extract_process.update_feature(name)
    if debug:
        feature = extract_process.rds_connector.download_features(name, return_type="dict")
        print(feature)
