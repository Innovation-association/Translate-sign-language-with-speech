#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time       : 2021/4/22 10:05
# @Author     : 代登辉
# @Email      : 3276336032@qq.com
# @File       : add1.py.py
# @Software   : PyCharm
# @Description: 方法描述,必填  
import numpy as np
import cv2

img=cv2.imread("1.jpg")
cv2.imshow("original",img)

#水平镜像
h_flip=cv2.flip(img,1)
cv2.imshow("Flipped Horizontally",h_flip)

#垂直镜像
v_flip=cv2.flip(img,0)
cv2.imshow("Flipped Vertically",v_flip)

#水平垂直镜像
hv_flip=cv2.flip(img,-1)
cv2.imshow("Flipped Horizontally & Vertically",hv_flip)


#平移矩阵[[1,0,-100],[0,1,-12]]
M=np.array([[1,0,-100],[0,1,-12]],dtype=np.float32)
img_change=cv2.warpAffine(img,M,(300,300))
cv2.imshow("translation",img_change)

#90度旋转
rows,cols=img.shape[:2]
M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("90",dst)

#45度旋转
rows,cols=img.shape[:2]
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("45",dst)

#缩放
height,width=img.shape[:2]
res=cv2.resize(img,(2*width,2*height))
cv2.imshow("large",res)

# 仿射变换
#对图像进行变换（三点得到一个变换矩阵）
# 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
# 然后再通过warpAffine来进行变换
point1=np.float32([[50,50],[300,50],[50,200]])
point2=np.float32([[10,100],[300,50],[100,250]])

M=cv2.getAffineTransform(point1,point2)
dst1=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,255,255))
cv2.imshow("affine transformation",dst1)
cv2.waitKey(0)