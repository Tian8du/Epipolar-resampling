# -*- encoding: utf-8 -*-
from osgeo import gdal
from osgeo import osr
import numpy as np
import time
import math
from Basefunction import point
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

if __name__ == '__main__':
    match_points = []
    count_above1 = 0
    leftimges = []
    rigtimages = []

    img11 = cv2.imread("K:\DSM\ZY3_test\L_epipolar_img.tiff", 2)
    img22 = cv2.imread("K:\DSM\ZY3_test\R_epipolar_img.tiff", 2)
    # img11 = cv2.imread("K:\\baihetanZY3\ZY302_TMS_E102.8_N27.1_20221126_L1A0001074472\DSM\L_epipolar_img.tiff",2)
    # img22 = cv2.imread("K:\\baihetanZY3\ZY302_TMS_E102.8_N27.1_20221126_L1A0001074472\DSM\R_epipolar_img.tiff",2)
    ss = 8
    for m in range(ss):
        for n in range(ss):
            row_step = int(img11.shape[0] / ss)
            line_step = int(img11.shape[1] / ss)

            img1 = img11[row_step*m:row_step*m+row_step, line_step*n:line_step*n+line_step]
            img2 = img22[row_step*m:row_step*m+row_step, line_step*n:line_step*n+line_step]
            # img2 = img22[0:5000, 0:5000]
            # img1 = img1.T
            # img2 = img2.T
            img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            sift = cv2.SIFT_create()

            # 检测特征点与描述符
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if (len(kp2) == 0 or len(kp1) == 0):
                leftimges.append(None)
                rigtimages.append(None)
                print("NUM", m, n, "None", "None")
                continue

            # 创建蛮力（BF）匹配器
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m1, m2) in enumerate(matches):
                if m1.distance < 0.3 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
                    matchesMask[i] = [1, 0]
                    pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
                    pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
                    # print(kpts1)
                    # print(i, pt1, pt2, )
                    leftimges.append(pt1)
                    rigtimages.append(pt2)
            # matches = sorted(matches, key=lambda x: x[0].distance)
            # if (len(matches) < 10 ):
            #     leftimges.append(None)
            #     rigtimages.append(None)
            #     print("NUM", m, n,"None", "None")
            # else:
            #     pt1 = kp1[matches[0][0].queryIdx].pt
            #     pt2 = kp2[matches[0][0].trainIdx].pt
            #     leftimges.append(pt1)
            #     rigtimages.append(pt2)
            #     print("NUM",m,n,"pt1",format(pt1[0], '.2f'), format(pt1[1], '.2f'),"pt2",
            #           format(pt2[0],'.2f'), format(pt2[1], '.2f'),"y_paral:",format(abs(pt2[1]-pt1[1]), '.2f'))

            sumy = 0
            pty = []
            count = 0
            for j in range(len(leftimges)):
                cut = abs(leftimges[j][1] - rigtimages[j][1])
                pty.append(cut)
                match_points.append(cut)
                if (cut > 1):
                    # print("Over", leftimges[j], rigtimages[j])
                    count = count + 1
                    count_above1 = count_above1 + 1

                sumy = sumy + abs(leftimges[j][1] - rigtimages[j][1])

            if (len(leftimges) == 0):
                ave_y = 0
            else:
                ave_y = sumy / len(leftimges)
            # print("chunk num",m+1, " ", n+1 )
            # print("ave_y", ave_y)
            # print("Max_ycut", max(pty))
            # print("Min_ycut", min(pty))
            # print("all_count", len(leftimges))
            # print("ycut>1:num", count, 'percent: {:.2%}'.format(count / len(leftimges)))
            if( len(pty) == 0):
                print("NUM", m, n, "Min_y_paral:", "None", "Ave_Y_paral:","None")
            else:
                print("NUM", m, n, "Min_y_paral:",format(min(pty), '.2f'),"Ave_Y_paral:",format(ave_y,'.2f'))
    # print("all_math_points",len(match_points))
    # print("ycut>1:num", count_above1, 'percent: {:.2%}'.format(count_above1 / len(match_points)))
    # sumy = 0
    # for i in range(len(match_points)):
    #     sumy = sumy + match_points[i]
    # ave = sumy / len(match_points)
    # print("ave y",ave)
    # mm = np.zeros((5,5,5), dtype=point)
    # mm[0,0,0] = point(2,3)
    # print(mm[0,0,0].x)




