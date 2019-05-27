import numpy as np
import cv2
import pylab as pl
import random


def BuildGaborKernels(ksize=5, lamda=1.5, sigma=1.0):
    """
    @description:构建gabor滤波器，生成多尺度，多方向的gabor特征
    @参数参考opencv
    @return:多个gabor卷积核所组成的
    """
    filters = []
    for theta in np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]):
        kern = cv2.getGaborKernel((ksize, ksize), sigma,
                                  theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        # gamma越大核函数图像越小，条纹数不变，sigma越大 条纹和图像都越大
        # psi这里接近0度以白条纹为中心，180度时以黑条纹为中心 psi这里取0.5
        # theta代表条纹旋转角度
        # lambd为波长 波长越大 条纹越大
        kern /= 1.5*kern.sum()
        filters.append(kern)
    # pl.figure(1)
    # for temp in range(len(filters)):
    #     pl.subplot(4, 4, temp + 1)
    #     pl.imshow(filters[temp], cmap='gray')
    # # pl.show()
    return filters


def GaborFeature(image):
    """
    @description:提取图像的gabor特征
    @image:灰度字符图像
    @return:滤波后的图
    """
    # retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    kernels = BuildGaborKernels(ksize=7, lamda=8, sigma=4)
    dst_imgs = []
    for kernel in kernels:
        img = np.zeros_like(image)
        tmp = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        img = np.maximum(img, tmp, img)
        dst_imgs.append(img)

    # pl.figure(2)
    # for temp in range(len(dst_imgs)):
    #     pl.subplot(4, 1, temp + 1)  # 第一个4为4个方向，第二个4为4个尺寸
    #     pl.imshow(dst_imgs[temp], cmap='gray')
    # pl.show()
    return dst_imgs


def GetImageFeatureGabor(image):
    """
    @description:提取经过Gabor滤波后字符图像的网格特征
    @image:普通有色彩图像
    @return:长度为32字符图像的特征向量feature
    """

    # 图像大小归一化
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h = image.shape[0]
    img_w = image.shape[1]

    # -----Gabor滤波--------------------------
    resImg = GaborFeature(image)
    # -----Gabor滤波--------------------------

    # -----对滤波后的图逐个网格化提取特征-------
    feature = np.zeros(64)  # 定义特征向量
    grid_size = 4
    imgcount = 0
    for img in resImg:
        # 二值化
        retval, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        imgcount += 1
        # pl.figure("dog")
        # pl.imshow(binary)
        # pl.show()

        # 计算网格大小
        grid_h = binary.shape[0] / grid_size
        grid_w = binary.shape[1] / grid_size
        for j in range(grid_size):
            for i in range(grid_size):
                # 统计每个网格中黑点的个数
                grid = binary[int(j * grid_h):int((j + 1) * grid_h), int(i * grid_w):int((i + 1) * grid_w)]
                feature[j * grid_size + i + (imgcount - 1) * grid_size * grid_size] = grid[grid == 0].size
        # print(feature)
        # featurelist = [y for x in feature for y in x]
    return feature.tolist()


if __name__ == '__main__':
    img = cv2.imread("plane.jpg")
    print(GetImageFeatureGabor(img))
    # print(GetImageFeatureGabor(img))
#     # --------------批量读图------------------------------------------
#     list = utils.eachFile(r"data/prepare_image")
#     count1 = 0
#     # 读图
#     ImgFeatures = []
#     ImgLabels = []
#     for filename in list:
#         # image = cv2.imread(filename, 0)
#         image = utils.cv_imread(filename, 0)
#         # 获取特征向量，传入灰度图
#         feature = GetImageFeatureGabor(image)
#         print(feature)
