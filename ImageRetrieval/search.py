import cv2
from PIL import Image
from pymongo import MongoClient
import numpy
from torch.autograd import Variable
from featureExtract.pHashFeature import *
from featureExtract.cannyFeature import *
from featureExtract.texturalGabor import *
from featureExtract.colorHSVFeature import *
from featureExtract.restFeature import DeepFeat
from torchvision import *
import torch
import pickle
import utils.CalcHammingRanking as CalcHR
import time


def computeEuclideanDistance(hash1, hash2):
    vec1 = np.array(hash1)
    vec2 = np.array(hash2)
    distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return distance


def searchPHashFeature(imgSelectedName):
    # 根据图像路径，读取图像
    img = cv2.imread(imgSelectedName)
    # 根据感知哈希算法，提取图像对应哈希特征
    hash = classify_pHash(img)
    lens = len(hash)
    photoColorFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # 文件名数组
        file = np.array(my_collection)
        # 特征值数组
        feature = np.array(cursor.count())
        # print(cursor.count())
        for item in cursor:
            dataColorHash = item["pHashFeature"]
            filenameColorHash = item["filename"]
            # print(dataColorHash)
            # print(filenameColorHash)
            cDistance = Hamming_distance(dataColorHash, hash)
            photoColorFeature[filenameColorHash] = cDistance
            # print(str(filenameColorHash) + ":" + str(cDistance))

        # 根据图像的特征距离，从低到高排序
        sortPhotoColorFeature = sorted(photoColorFeature.items(), key=lambda item: item[1])
        # print(sortPhotoColorFeature[0:9])

        # 取前九个距离最近的图像，将文件名和特征距离保存在列表中
        imgColorNames = []
        imgColorRates = []
        for photoColorFeature in sortPhotoColorFeature[0:9]:
            # print(photoColorFeature)
            imgColorNames.append("data/prepare_image/" + str(photoColorFeature[0]))
            imgColorRates.append((lens - photoColorFeature[1]) / lens)
        # for photoColorFeature in sortPhotoColorFeature[0:9]:
        # print(photoColorFeature)
        # imgColorNames.append(str(photoColorFeature[1]))
        # print(imgColorNames)
        return imgColorNames, imgColorRates
    except Exception as e:
        print(e)


# def match(feature1, feature2):
#     # 计算欧式距离
#     return torch.pow(feature1 - feature2, 2).sum().item()

def searchRestFeature(imgSelectedName):
    deep = DeepFeat()
    # 根据图像路径，读取图像
    # img = cv2.imread(imgSelectedName)
    # 提取深度学习特征
    t1 = time.time()
    restFeature = deep(imgSelectedName)
    t2 = time.time()
    print("feature extract time:")
    print((t2-t1)*1000)
    photoRestFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # 文件名数组
        file = np.array(my_collection)
        # 特征值数组
        feature = np.array(cursor.count())
        # print(cursor.count())
        for item in cursor:
            dataRestHash = pickle.loads(item["restFeature"])
            # print(dataRestHash)
            filenameRestHash = item["filename"]
            # print(dataColorHash)
            # print(filenameColorHash)
            cDistance = deep.match(restFeature, dataRestHash)
            photoRestFeature[filenameRestHash] = cDistance
            # print(str(filenameColorHash) + ":" + str(cDistance))

        # 根据图像的特征距离，从低到高排序
        sortPhotoRestFeature = sorted(photoRestFeature.items(), key=lambda item: item[1])
        # print(sortPhotoColorFeature[0:9])
        max1 = max(sortPhotoRestFeature[:-1])
        print(max1[1])
        # 取前九个距离最近的图像，将文件名和特征距离保存在列表中
        imgColorNames = []
        imgColorRates = []
        for photoRestFeature in sortPhotoRestFeature[0:9]:
            # print(photoColorFeature)
            imgColorNames.append("data/prepare_image/" + str(photoRestFeature[0]))
            imgColorRates.append(round((30000-photoRestFeature[1])/30000,2))
        # for photoColorFeature in sortPhotoColorFeature[0:9]:
        # print(photoColorFeature)
        # imgColorNames.append(str(photoColorFeature[1]))
        # print(imgColorNames)
        return imgColorNames, imgColorRates
    except Exception as e:
        print(e)


def searchTexturalFeature(imgSelectedName):
    # 根据图像路径，读取图像
    img = cv2.imread(imgSelectedName)
    # 根据感知哈希算法，获取滤波特征
    gaborFeature = GetImageFeatureGabor(img)
    photoGaborFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # 文件名数组
        file = np.array(my_collection)
        # 特征值数组
        feature = np.array(cursor.count())
        # print(cursor.count())
        for item in cursor:
            dataGaborHash = item["gaborFeature"]
            filenameColorHash = item["filename"]
            # print(dataColorHash)
            # print(filenameColorHash)
            cDistance = computeEuclideanDistance(dataGaborHash, gaborFeature)
            photoGaborFeature[filenameColorHash] = cDistance
            # print(str(filenameColorHash) + ":" + str(cDistance))

        # 根据图像的特征距离，从低到高排序
        sortPhotoGaborFeature = sorted(photoGaborFeature.items(), key=lambda item: item[1])
        # print(sortPhotoColorFeature[0:9])

        # 取前九个距离最近的图像，将文件名保存在列表中
        imgGaborNames = []
        for photoGaborFeature in sortPhotoGaborFeature[0:9]:
            # print(photoGaborFeature)
            imgGaborNames.append("data/prepare_image/" + str(photoGaborFeature[0]))
        for photoGaborFeature in sortPhotoGaborFeature[0:9]:
            # print(photoGaborFeature)
            imgGaborNames.append(str(photoGaborFeature[1]))
        return imgGaborNames
    except Exception as e:
        print(e)


def searchCannyFeature(imgSelectedName):
    # 根据图像路径，读取图像
    imgCanny = cv2.imread(imgSelectedName)
    #
    # hash = classify_pHash(img)
    t1 = time.time()
    cannyFeature = GetCannyFeature(imgCanny)
    t2 = time.time()
    print("feature extract time:")
    print((t2 - t1) * 1000)
    photoCannyFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # 文件名数组
        file = np.array(my_collection)
        # 特征值数组
        feature = np.array(cursor.count())
        # print(cursor.count())
        for item in cursor:
            dataCannyHash = item["cannyFeature"]
            filenameCannyHash = item["filename"]
            # print(dataColorHash)
            # print(filenameColorHash)
            # cDistance = Hamming_distance(dataCannyHash, cannyFeature)
            cDistance = computeEuclideanDistance(dataCannyHash, cannyFeature)
            photoCannyFeature[filenameCannyHash] = cDistance

        # 根据图像的特征距离，从低到高排序
        sortPhotoCannyFeature = sorted(photoCannyFeature.items(), key=lambda item: item[1])
        # print(sortPhotoColorFeature[0:9])
        max1 = max(sortPhotoCannyFeature[:-1])
        # 取前九个距离最近的图像，将文件名保存在列表中
        imgCannyNames = []
        imgCannyDis = []
        for CannyFeature in sortPhotoCannyFeature[0:9]:
            # print(CannyFeature)
            imgCannyNames.append("data/prepare_image/" + str(CannyFeature[0]))
            imgCannyDis.append(1 - CannyFeature[1] / max1[1])
        # for CannyFeature in sortPhotoCannyFeature[0:9]:
        #     # print(CannyFeature)
        #     imgCannyDis.append(1-int(CannyFeature[1])/max1)
        # print(imgColorNames)
        return imgCannyNames, imgCannyDis
    except Exception as e:
        print(e)


def searchGaborFeature(imgSelectedName):
    # 根据图像路径，读取图像
    imgGabor = cv2.imread(imgSelectedName)
    #
    # hash = classify_pHash(img)
    t1 = time.time()
    nowGaborFeature = GetImageFeatureGabor(imgGabor)
    t2 = time.time()
    print("feature extract time:")
    print((t2 - t1) * 1000)
    photoGaborFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # # 文件名数组
        # file = numpy.array(my_collection)
        # # 特征值数组
        # feature = numpy.array(cursor.count())

        # print(cursor.count())
        for item in cursor:
            dataGaborHash = item["gaborFeature"]
            filenameGaborHash = item["filename"]
            # print(dataColorHash)
            # print(filenameColorHash)
            # cDistance = Hamming_distance(dataCannyHash, cannyFeature)
            cDistance = computeEuclideanDistance(dataGaborHash, nowGaborFeature)
            photoGaborFeature[filenameGaborHash] = cDistance

        # 根据图像的特征距离，从低到高排序
        sortPhotoGaborFeature = sorted(photoGaborFeature.items(), key=lambda item: item[1])
        # print(sortPhotoColorFeature[0:9])
        print("max")
        max1 = max(sortPhotoGaborFeature[:-1])
        print(max1)
        # 取前九个距离最近的图像，将文件名保存在列表中
        imgGaborNames = []  # 图像名
        imgRate = []  # 置信度
        for GaborFeaturea in sortPhotoGaborFeature[0:9]:
            # print(CannyFeature)
            imgGaborNames.append("data/prepare_image/" + str(GaborFeaturea[0]))
            # imgRate.append((1-GaborFeaturea[1])/max1[1])
            imgRate.append(1 - (GaborFeaturea[1] / max1[1]))
        # for CannyFeature in sortPhotoGaborFeature[0:9]:
        # print(CannyFeature)
        # imgGaborNames.append(str(CannyFeature[1]))
        # print(imgColorNames)
        return imgGaborNames, imgRate
    except Exception as e:
        print(e)


def searchHSVColorFeature(imgSelectedName):
    # 根据图像路径，读取图像
    imgHSV = cv2.imread(imgSelectedName)

    # 根据颜色直方图算法，提取图像对应哈希特征
    t1 = time.time()
    HSVFeature = GetHSVFeature(imgHSV)
    t2 = time.time()
    print("feature extract time:")
    print((t2 - t1) * 1000)
    # print("hash算法为")
    # print(HSVFeature)

    photoHSVFeature = {}  # 初始化一个颜色特征字典
    # 遍历数据库，查询图像颜色特征并比较
    try:
        # 连接数据库
        client = MongoClient(host="localhost", port=27017)
        db = client.retrieval
        my_collection = db.imgInfo

        # 遍历集合
        cursor = my_collection.find()
        # 文件名数组
        file = np.array(my_collection)
        # 特征值数组
        feature = np.array(cursor.count())
        # print(cursor.count())
        for item in cursor:
            # dataCannyHash = item["colorFeature"]
            dataHSVFeature = pickle.loads(item["colorFeature"])
            filenameCannyHash = item["filename"]
            # print(dataHSVFeature)
            # 关联度越大越相似
            # print(dataHSVFeature)
            cDistance = cv2.compareHist(dataHSVFeature, HSVFeature, cv2.HISTCMP_CORREL)
            # print(cDistance)
            # cDistance = computeEuclideanDistance(dataCannyHash, cannyFeature)
            photoHSVFeature[filenameCannyHash] = cDistance

        # 相关度越高，越相似，将相关度按照从大到小排序
        sortPhotoHSVFeature = sorted(photoHSVFeature.items(), key=lambda item: item[1], reverse=True)
        # print(sortPhotoColorFeature[0:9])

        # 取前九个距离最近的图像，将文件名保存在列表中
        imgHSVNames = []
        imgHSVNums = []
        for HSVFeature in sortPhotoHSVFeature[0:9]:
            # print(CannyFeature)
            imgHSVNames.append("data/prepare_image/" + str(HSVFeature[0]))
            imgHSVNums.append(HSVFeature[1])
        # print(imgColorNames)
        return imgHSVNames, imgHSVNums
    except Exception as e:
        print(e)


def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = numpy.zeros([num_data, bit], dtype=numpy.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else:
            data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B


def _GenerateCode(model, img, bit, use_gpu):
    B = numpy.zeros([1, bit], dtype=numpy.float32)
    data_input = img
    if use_gpu:
        data_input = Variable(data_input.cuda())
    else:
        data_input = Variable(data_input)
    output = model(data_input)
    if use_gpu:
        B[0, :] = torch.sign(output.cpu().data).numpy()
    else:
        B[0, :] = torch.sign(output.data).numpy()
    return B


def searchDeepFeature(bit, imgSelectedName):
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # use model
    modelname = 'log/DPSH_48bits_CIFAR-10_19-03-20-07-51-32.pkl'
    model = torch.load(modelname, map_location='cpu')

    model.eval()
    img = Image.open(imgSelectedName)
    t1 = time.time()
    img = img.convert('RGB')
    img = transformations(img).unsqueeze(0)
    _qB = _GenerateCode(model, img, bit, 0)
    t2 = time.time()
    print("feature extract time:")
    print((t2 - t1) * 1000)
    dB = numpy.load('log/dB_' + str(bit) + 'bits.npy')
    imagelist, a = CalcHR._CalcMap(_qB, dB, 9)
    imgDeepNames = []
    imgDistance = []
    for d in a:
        imgDistance.append((100 - d)/100)
    for imgDeepName in imagelist:
        # print("come in")
        imgDeepNames.append('data/CIFAR-10/database_img/' + str(imgDeepName + 1) + '.png')
    # print(imgDeepNames)
    return imgDeepNames, imgDistance


if __name__ == '__main__':
    img, a = searchGaborFeature('data/prepare_image/3.png')
    print(img)
    print(a)
# for i in
# hamingdistans = Hamming_distance(hash)
