"""
hist = cv2.calcHist([image],             # 传入图像（列表）
                    [0],                 # 使用的通道（使用通道：可选[0],[1],[2]）
                    None,                # 没有使用mask(蒙版)
                    [256],               # HistSize
                    [0.0,255.0])         # 直方图柱的范围
                                         # return->list
"""

import cv2
import numpy as np

def GetHSVFeature(image):
    # hist1 = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV", HSV)
    HSVhist = cv2.calcHist([HSV], [0], None, [256], [0.0, 255.0])
    cv2.normalize(HSVhist, HSVhist, 0, 255 * 0.9, cv2.NORM_MINMAX)
    # cv2.imshow("HSVhist", HSVhist)
    return HSVhist


# def GetRGBFeature(image):
#     # hist1 = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
#     RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     RGBhist = cv2.calcHist([RGB], [0], None, [256], [0.0, 255.0])
#     cv2.normalize(RGBhist, RGBhist, 0, 255 * 0.9, cv2.NORM_MINMAX)
#     return RGBhist


# def GetBRGColorFeature():
#     img = cv2.imread('test.png')
#     h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像
#
#     bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
#     color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
#     for ch, col in enumerate(color):
#         originHist = cv2.calcHist([img], [ch], None, [256], [0, 256])
#         cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
#         hist = np.int32(np.around(originHist))
#         pts = np.column_stack((bins, hist))
#         cv2.polylines(h, [pts], False, col)
#
#     h = np.flipud(h)
#     print(h)
#
#     cv2.imshow('colorhist', h)
#     cv2.waitKey(0)


if __name__ == '__main__':
    image1 = cv2.imread('plane.jpg')
    # image2 = cv2.imread('dog1.png')
    # image3 = cv2.imread('test.png')
    # image = cv2.imread('8.png')
    # hist1 = GetHSVFeature(image1)
    # hist = GetHSVFeature(image1)
    # GetBRGColorFeature()
    # hist2 = GetHSVFeature(image2)
    # hist3 = GetHSVFeature(image3)
    # print(hist)
    # plt.subplot(2, 1, 1)
    # plt.plot(hist1)
    # plt.subplot(2, 1, 2)
    # plt.plot(hist2)
    # print(hist1)
    # plt.subplot(2, 1, 3)
    # plt.plot(hist3)

    # plt.show()
    # retval1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # retval2 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CORREL)
    # print(retval1)
    # print(retval2)
    # GetHSVFeature()
    # GetBRGColorFeature()
    #
    # histImgB = calcAndDrawHist(b, [255, 0, 0])
    # histImgG = calcAndDrawHist(g, [0, 255, 0])
    # histImgR = calcAndDrawHist(r, [0, 0, 255])
    #
    # cv2.imshow("histImgB", histImgB)
    # cv2.imshow("histImgG", histImgG)
    # cv2.imshow("histImgR", histImgR)
    # cv2.imshow("Img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
