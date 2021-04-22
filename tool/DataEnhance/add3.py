#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time       : 2021/4/22 13:28
# @Author     : 代登辉
# @Email      : 3276336032@qq.com
# @File       : add3.py
# @Software   : PyCharm
# @Description: 读取文件夹下数据
import os


def traverse(f):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            print('文件: %s' % tmp_path)
        else:
            print('文件夹：%s' % tmp_path)
            traverse(tmp_path)


path = '..\\..\\train\data\\trainData\\test001\\'
traverse(path)