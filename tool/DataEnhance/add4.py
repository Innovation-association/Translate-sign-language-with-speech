#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time       : 2021/4/22 13:39
# @Author     : 代登辉
# @Email      : 3276336032@qq.com
# @File       : add4.py
# @Software   : PyCharm
# @Description: 数据增强

import os
import numpy as np
import cv2


save_path = "../../train/data/trainData/test002/"


def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# dimming
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def img_augmentation(path, name_int):
    img = cv2.imread(path)

    img_flip = cv2.flip(img, 1)  # flip
    img_rotation = rotate(img)  # rotation

    img_noise1 = SaltAndPepper(img, 0.3)
    img_noise2 = addGaussianNoise(img, 0.3)

    img_brighter = brighter(img)
    img_darker = darker(img)


    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    dst_30 = cv2.warpAffine(img, M, (cols, rows))

    img_gasuss = gasuss_noise(img)

    cv2.imwrite(save_path + "add" + '%s' % str(name_int) + '.jpg', img_flip)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 1) + '.jpg', img_rotation)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 2) + '.jpg', img_noise1)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 3) + '.jpg', img_noise2)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 4) + '.jpg', img_brighter)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 5) + '.jpg', img_darker)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 6) + '.jpg', dst_30)
    cv2.imwrite(save_path + "add" + '%s' % str(name_int + 7) + '.jpg', img_gasuss)
    print("end")


def traverse(f):
    fs = os.listdir(f)
    i=1
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            print(tmp_path)
            img_augmentation(tmp_path, i)
        i = i+8

path = '..\\..\\train\data\\trainData\\test001\\'
traverse(path)