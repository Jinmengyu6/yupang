# 边缘检测算法
# coding=utf-8
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import *


def GetCannyFeature(img):
    # canny边缘检测算法
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化图像
    # cv2.imshow("gray", grayimg)
    img1 = cv2.GaussianBlur(grayimg, (5, 5), 0)  # 高斯平滑处理 图片降噪
    # cv2.imshow("Gauss", img1)
    canny1 = cv2.Canny(img1, 50, 120)  # 读取canny算子边缘提取
    # cv2.imshow("edge1", canny1)

    # print(newX)
    newX_scaler = MinMaxScaler()
    newX_train = newX_scaler.fit_transform(canny1)    # 归一化处理

    # 利用pca做降维处理
    pca = PCA(n_components=32)
    newX = pca.fit_transform(newX_train)
    newX = [y for x in newX for y in x]
    # print(newX)
    # canny = [y for x in newX_train for y in x]
    # print(canny)
    # print(canny)
    # print(pca.explained_variance_ratio_)  # 判断每维特征能够表示的
    # cv2.waitKey(0)
    return newX


def computeEuclideanDistance(hash1, hash2):
    vec1 = np.array(hash1)
    vec2 = np.array(hash2)
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return distance


if __name__ == '__main__':
    img1 = cv2.imread("timg.jpg")
    l1 = GetCannyFeature(img1)
