import cv2
from sklearn.cluster import KMeans
import sys
import os


def GetSurfFeature(img):
    # canny边缘检测算法
    # print(img)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.GaussianBlur(grayimg, (3, 3), 0)
    # canny1 = cv2.Canny(img1, 50, 120)
    detector = cv2.xfeatures2d.SURF_create(2000)
    kps, des = detector.detectAndCompute(grayimg, None)
    """
    16 sift对象会使用DoG检测关键点，对关键点周围的区域计算向量特征，检测并计算
    17 返回 关键点和描述符
    18 关键点是点的列表
    19 描述符是检测到的特征的局部区域图像列表
    20 
    21 关键点的属性：
    22     pt: 点的x y坐标
    23     size： 表示特征的直径
    24     angle: 特征方向
    25     response: 关键点的强度
    26     octave: 特征所在金字塔层级
    27         算法进行迭代的时候， 作为参数的图像尺寸和相邻像素会发生变化
    28         octave属性表示检测到关键点所在的层级
    29     ID： 检测到关键点的ID
    30 
     """
    cv2.drawKeypoints(image=img, outImage=img,
                      keypoints=kps, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(255, 0, 0))
    cv2.imshow("surf_detected", img)
    print(des)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(pca.explained_variance_ratio_)  判断每维特征能够表示的


if __name__ == '__main__':
    # img = cv2.imread("img-49004-airplane.png")
    img = cv2.imread("timg.jpg")
    GetSurfFeature(img)
