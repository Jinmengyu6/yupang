from pymongo import *
import pickle
import os
from bson.binary import Binary
from featureExtract.pHashFeature import *
from featureExtract.cannyFeature import *
from featureExtract.texturalGabor import *
from featureExtract.colorHSVFeature import *
from featureExtract.restFeature import DeepFeat
import numpy as np


# def insert(filename):
#     try:
#         client = MongoClient(host="localhost", port=27017)
#         db = client.retrieval
#         db.ps.insert_one({"filename": filename})
#     except Exception as e:
#         print(e)


def saveFeature():
    deep = DeepFeat()
    for file in os.listdir("../data/prepare_image"):
        # print(str(file))
        img = cv2.imread("../data/prepare_image/" + str(file))
        # print(img)
        # feature = getHash(img)
        # print(feature)
        phash_feature = classify_pHash(img)
        canny_feature = GetCannyFeature(img)
        # print(canny_feature[0])
        gabor_feature = GetImageFeatureGabor(img)
        HSV_feature = GetHSVFeature(img)
        restFeature = deep("../data/prepare_image/" + str(file))
        # print(HSV_feature)
        print(restFeature)
        # print("color=" + colorFeature+ "canny=" + cannyFeature + "gabor=" + gaborFeature)
        # print(colorFeature)
        # print(cannyFeature)
        # print(gaborFeature)
        try:
            client = MongoClient(host="localhost", port=27017)
            # db = client.demo
            # db.stu.insert_one({"name": "wjw", "age": 20})
            db = client.retrieval
            db.imgInfo.insert_one({"filename": str(file),
                                   "colorFeature": Binary(pickle.dumps(HSV_feature, protocol=-1)),
                                   "cannyFeature": canny_feature,
                                   "gaborFeature": gabor_feature,
                                   "pHashFeature": phash_feature,
                                   "restFeature":Binary(pickle.dumps(restFeature, protocol=-1))})
            # })
        except Exception as e:
            print(e)


if __name__ == '__main__':
    saveFeature()
