import cv2
import numpy as np
import os


# 可以读取带中文路径的图
def cv_imread(file_path, type=0):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if type == 0:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img


# 扫描输出图片列表
def eachFile(filepath):
    list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list
