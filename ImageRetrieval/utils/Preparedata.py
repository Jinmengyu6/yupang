import pickle
import cv2
import numpy as np


# def load(filename):
# #     with open(filename, 'rb') as fo:
# #         data = pickle.load(fo, encoding='latin1')
# #     return data
# #
# # p = 'cifar-10-batches-py/data_batch_1'
# # d = load(p)
# # print(d.keys())
# # print(d['batch_label'])
# # print(d['labels'])
# # print(d['data'])
# # print(d['filenames'])
# #
# # e = d['data']
# # for i in range(len(e)):
# #     cv2.imwrite('a\\'+str(i)+'.jpg',
# #                 e[i].reshape(32, 32, 3))
# # print("done!")

import numpy as np
from PIL import Image
import pickle
import os

CHANNEL = 3
WIDTH = 32
HEIGHT = 32


data = []
labels = []
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(5):
    with open("data/cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data += list(data_dict[b'data'])
        labels += list(data_dict[b'labels'])

img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

data_path = "../data/images/"
if not os.path.exists(data_path):
    os.makedirs(data_path)
for i in range(img.shape[0]):
    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    name = "img-" + str(i) + "-" + classification[labels[i]] + ".png"
    rgb.save(data_path + name, "PNG")
    print("done!")




