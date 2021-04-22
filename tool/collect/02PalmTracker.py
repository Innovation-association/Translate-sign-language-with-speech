#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time       : 2021/4/22 10:05
# @Author     : 代登辉
# @Email      : 3276336032@qq.com
# @File       : 02PalmTracker.py
# @Software   : PyCharm
# @Description: 数据收集 按S保存q推出 正常

import argparse
import cv2

bg = None

# 数据存储路径
save_path = '../../train/data/trainData/gestures/Under'

# 截取矩形框的坐标
top, right, bottom, left = 90, 380, 285, 590
lineType = 5  # 矩形框宽度
point_color = (0, 0, 255)  # 矩形框为颜色

ptLeftTop = (left + lineType, top - lineType)
ptRightBottom = (right - lineType, bottom + lineType)


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (contours, cnts, _) = cv2.findContours(thresholded.copy(),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main(dtype):
    camera = cv2.VideoCapture(0)

    num_frames = 0
    thresholded = None

    count = 0

    while (True):
        (grabbed, frame) = camera.read()
        if grabbed:

            frame = cv2.flip(frame, 1)

            cv2.rectangle(frame, ptLeftTop, ptRightBottom, point_color, lineType)

            clone = frame[top:bottom, right:left]  # x,x+w y y+w
            cv2.imshow('Video Feed', clone)

            cv2.imshow('Thesholded', frame)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord('q'):
                break

            if keypress == ord('s'):
                print(save_path + '/{:04}.jpg'.format(count))
                path = save_path + '/{:04}.jpg'.format(count)
                cv2.imwrite(path, clone)
                count += 1
                print(count, 'saved.')
                num_frames += 1
        else:
            camera.release()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='pause')
    args = parser.parse_args()
    main(args.dtype)
    cv2.destroyAllWindows()
