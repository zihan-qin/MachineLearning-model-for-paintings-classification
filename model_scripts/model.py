# !/usr/bin/python
# -*- coding:utf-8 -*-
# File Name: train_model.py
# Author: Zihan Qin
# Mail: zihanqin@usc.edu
# Created Time: 2021-11-23 23:41:06


import numpy as np
import model_utils
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import svm
from sklearn import model_selection

class TrainModel:
    def __init__(self, rds_info):
        raw_data = model_utils.load_data(rds_info)
        data = model_utils.raw_2_sample(raw_data)
        self.x = np.array([each.feature for each in data])
        self.y_artist = np.array([each.y_artist for each in data])
        self.y_genre = np.array([each.y_genre for each in data])
        self.y_style = np.array([each.y_style for each in data])
        self.s3_session = model_utils.S3ModelSession()

    def train_model(self, x, y, local_model_path, model_name):
        clf = pipeline.Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('svm', svm.SVC(kernel='rbf'))]
        )
        param = {
            "svm__C": [0.1, 1, 10],
            "svm__gamma": [0.01, 0.05, 0.1]
        }
        grid = model_selection.GridSearchCV(clf, param_grid = param, cv=5, scoring="f1_weighted")
        grid.fit(x, y)
        print("best_score: %.3f\n" % grid.best_score_)
        best_model = grid.best_estimator_.fit(x, y)
        self.s3_session.upload_model(best_model, local_model_path, model_name)

    def create_model(self, task="genre", local_model_path=None):
        print("create model for task %s\n" % task)
        model_name = "%s_model.jl" % task
        self.train_model(self.x, eval("self.y_%s" % task), local_model_path, model_name)


class PredictModel:
    def __init__(self, rds_info, local_model_path=None):
        self.rds_info = rds_info
        self.local_model_path = local_model_path
        self.tasks = ("genre", "artist", "style")
        self.models = {}
        self.s3_session = model_utils.S3ModelSession()

    def init_model(self, task):
        """task: genre/artist/style"""
        model_name = "%s_model.jl" % task
        self.models[task] = self.s3_session.download_model(model_name, self.local_model_path)

    def predict(self, task, name):
        """task: genre/artist/style"""
        if task not in self.models:
            self.init_model(task)
        model = self.models[task]
        raw_data = model_utils.load_pred_data(rds_info, name)
        feature = np.array(model_utils.raw_2_sample(raw_data, train=False)[0].feature).reshape(1, -1)
        pred = model.predict(feature)[0]
        return pred


if __name__ == "__main__":
    debug_type = "pred"

    rds_info = {
        'host': "wikiartdb.cv8worynfzsx.us-west-1.rds.amazonaws.com",
        'user': "admin",
        'password': "wikiartdb",
        'db': "wikiart",
        'port': 3306
    }

    if debug_type == "train":
        train_process = TrainModel(rds_info)
        genre_model_path = train_process.create_model(task="genre")
        genre_model_path = train_process.create_model(task="artist")
        genre_model_path = train_process.create_model(task="style")
    elif debug_type == "pred":
        name = "albrecht-durer_abduction"
        predict_model = PredictModel(rds_info)
        print(predict_model.predict("genre", name))
        print(predict_model.predict("style", name))
        print(predict_model.predict("artist", name))
    else:
        raise NotImplementedError("debug type %s does not exist" % debug_type)
