# !/usr/bin/python
# -*- coding:utf-8 -*-
# File Name: image_process.py
# Author: Zihan Qin
# Mail: zihanqin@usc.edu
# Created Time: 2021-10-28 17:05:25


import cv2
import leargist
import numpy as np
from PIL import Image


def sift(image, vector_size=32):
    try:
        alg = cv2.SIFT_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print(f"Error: {e}")
        return None
    return dsc


def color_moments(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    color_feature = []
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    color_feature.extend([h_std, s_std, v_std])
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature

def color_gist(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    descriptor = leargist.color_gist(image)
    return descriptor
