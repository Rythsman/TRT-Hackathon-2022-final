import os
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
import onnxruntime as ort

npz_file = "image_0_preprocess.npz"

def testProcess():
    weight = np.load(npz_file)
    # print(weight.files)
    img = weight['data']
    img = img.squeeze(0).transpose(1, 2, 0)
    img = img*255
    img = np.floor(img) # ceil 向上取整
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("tmp.png", img)
    # plt.figure()
    # plt.imshow("tmp.png", img)


def testOrt():
    