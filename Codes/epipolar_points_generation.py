import numpy as np
import os
import rpc
from osgeo import gdal
from Basefunction import point
from Basefunction import cal_coff_aff3
from Basefunction import cal_position
import Basefunction
import relocate
import math
import matplotlib.pyplot as plt
import time
import Polytran_and_resample
from filesave import Save_as_tif
from Polytran_and_resample import cal_chunk
import multiprocessing
from Parallel import cal_chunks
from Coordinate import Cal_epipolar_geotrans_L
from Coordinate import Cal_epipolar_geotrans_R


# Step1 one point from LeftImage to rightImage
def Left2Right1(rpc1, rpc2, initial, h_Min, h_Max):
    '''
    Central Points in left image ----> Geodetic coordinates in Max and Min height  by Negative_RPCModle
    Geodetic coordinates in Max and Min height --->  Corresponding two Projection points in right image
    :param rpc1: the rpc of left image
    :param rpc2: the rpc of right image
    :param x: the value of  Left central Point's X
    :param y: the value of  Left central Point's X
    :param h_Min: the height of Minimum
    :param h_Max: the height of Maximum
    :return: Corresponding two Projection points in right image
    '''
    temp_point1 = point()
    temp_point2 = point()
    lon1, lat1 = rpc.RPC_Negative(initial.y, initial.x, h_Min, rpc1)
    lon2, lat2 = rpc.RPC_Negative(initial.y, initial.x, h_Max, rpc1)
    temp_point1.y, temp_point1.x = rpc.RPC_Positive(lon1, lat1, h_Min, rpc2)
    temp_point2.y, temp_point2.x = rpc.RPC_Positive(lon2, lat2, h_Max, rpc2)
    return temp_point1, temp_point2


# Step2: two points(RightImage)--->>> two Points(LeftImage)
def Right2Left(rpc1, rpc2, point1, point2, h_Min, h_Max):
    '''
    Two corresponding points in right image ---> Geodetic coordinates in Max and Min height  by Negative_RPCModle
    Geodetic coordinates in Max and Min height(two points) --->  Corresponding two Projection points in left image
    :param rpc1: the rpc of left image
    :param rpc2: the rpc of right image
    :param point1: one point in right image
    :param point2: another point in right image
    :param h_Min: the height of Minimum
    :param h_Max: the height of Maximum
    :return: Corresponding two points in left image
    '''
    temp_point1 = point()
    temp_point2 = point()
    lon1, lat1 = rpc.RPC_Negative(point1.y, point1.x, h_Max, rpc2)
    lon2, lat2 = rpc.RPC_Negative(point2.y, point2.x, h_Min, rpc2)
    temp_point1.y, temp_point1.x = rpc.RPC_Positive(lon1, lat1, h_Max, rpc1)
    temp_point2.y, temp_point2.x = rpc.RPC_Positive(lon2, lat2, h_Min, rpc1)
    return temp_point1, temp_point2


# Step3:two points(LeftImage)--->>> two points(RightImage)
def Left2Right2(rpc1, rpc2, point1, point2, h_Min, h_Max):
    '''
    Two points in left image ---> Geodetic coordinates in Max and Min height by Negative_PRCModel
    Geodetic coordinates in Max and Min height(two points) --->  Corresponding two Projection points in left image
    :param rpc1: the rpc of left image
    :param rpc2: the rpc of right image
    :param point1: one point in left image
    :param point2: another point in left image
    :param h_Min: the height of Minimum
    :param h_Max: the height of Maximum
    :return: Corresponding tow points in right image
    '''
    temp_point1 = point()
    temp_point2 = point()
    lon1, lat1 = rpc.RPC_Negative(point1.y, point1.x, h_Min, rpc1)
    lon2, lat2 = rpc.RPC_Negative(point2.y, point2.x, h_Max, rpc1)
    temp_point1.y, temp_point1.x = rpc.RPC_Positive(lon1, lat1, h_Min, rpc2)
    temp_point2.y, temp_point2.x = rpc.RPC_Positive(lon2, lat2, h_Max, rpc2)
    return temp_point1, temp_point2


# another step: get the start point of Right image
def get_Rstartpoint(rpc1, rpc2, initial_point, h):
    '''
    one point in the y axies of left image ---> one Geodetic point in fixed height by Negative_RPCModel
    ---> corresponding point in right image by Positive_RPCModel
    :param rpc1: the rpc of left image
    :param rpc2: the rpc of right image
    :param initial_point: initial point in left image
    :param h: the fixed height
    :return: the corresponding point in right image
    '''
    temp_point = point()
    lon, lat = rpc.RPC_Negative(initial_point.y, initial_point.x, h, rpc1)
    temp_point.y, temp_point.x = rpc.RPC_Positive(lon, lat, h, rpc2)
    return temp_point


def get_R_points(rpc1, rpc2, L_points, h):
    R_points = []
    for i in range(len(L_points)):
        pt = point()
        pt = get_Rstartpoint(rpc1, rpc2, L_points[i], h)
        R_points.append(pt)
    return R_points


def RPC_process(rpc1, rpc2, x, y, h_Min, h_Max, h_constant, dataL, dataR, num):
    '''
    get points in image
    :param rpc1:
    :param rpc2:
    :param x:
    :param y:
    :param h_Min:
    :param h_Max:
    :param h_constant:
    :return:
    '''
    pointsL = []
    pointsR = []
    RR_start_points = []
    ini_point = point()
    ini_point.x = x
    ini_point.y = y
    if cal_position(ini_point, dataL):
        pointsL.append(ini_point)
    RR_start_point = get_Rstartpoint(rpc1, rpc2, ini_point, h_constant)
    if cal_position(RR_start_point, dataR):
        RR_start_points.append(RR_start_point)
    point11, point22 = Left2Right1(rpc1, rpc2, ini_point, h_Min, h_Max)
    # if (cal_position(point11, dataR) == 0 or cal_position(point22, dataR) == 0):
    #     return None, None, None
    # else:
    #     pointsR.append(point11)
    #     pointsR.append(point22)
    if cal_position(point11, dataR):
        pointsR.append(point11)
    if cal_position(point22, dataR):
        pointsR.append(point22)

    point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
    # if (cal_position(point11, dataL) == 0 and cal_position(point22, dataL) == 0):
    #     return pointsL, pointsR, RR_start_points
    # else:
    if cal_position(point11, dataL):
        pointsL.append(point11)
    if cal_position(point22, dataL):
        pointsL.append(point22)

    # while ( cal_position(point11, dataL) or cal_position(point22, dataL)): a, b = point11, point22 while (
    # cal_position(point11, dataL) or cal_position(point22, dataL) or cal_position(a, dataR) or cal_position(b, dataR)):
    for i in range(num):
        point11, point22 = Left2Right2(rpc1, rpc2, point11, point22, h_Min, h_Max)
        if cal_position(point11, dataR):
            pointsR.append(point11)
        if cal_position(point22, dataR):
            pointsR.append(point22)
        point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
        if cal_position(point11, dataL):
            pointsL.append(point11)
        if cal_position(point22, dataL):
            pointsL.append(point22)
    if len(pointsL) == 0 or len(pointsR) == 0:
        return None, None, None

    return pointsL, pointsR, RR_start_points


def RPC_process2(rpc1, rpc2, x, y, h_Min, h_Max, h_constant, dataL, dataR, num):
    '''

    :param rpc1:
    :param rpc2:
    :param x:
    :param y:
    :param h_Min:
    :param h_Max:
    :param h_constant:
    :param dataL:
    :param dataR:
    :param num: 左右循环次数
    :return:
    '''
    pointsL = []
    pointsR = []
    RR_start_points = []
    ini_point = point()
    ini_point.x = x
    ini_point.y = y

    pointsL.append(ini_point)
    RR_start_point = get_Rstartpoint(rpc1, rpc2, ini_point, h_constant)

    RR_start_points.append(RR_start_point)
    point11, point22 = Left2Right1(rpc1, rpc2, ini_point, h_Min, h_Max)
    pointsR.append(point11)
    pointsR.append(point22)

    point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
    pointsL.append(point11)
    pointsL.append(point22)

    for i in range(num):
        point11, point22 = Left2Right2(rpc1, rpc2, point11, point22, h_Min, h_Max)
        pointsR.append(point11)
        pointsR.append(point22)
        point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
        pointsL.append(point11)
        pointsL.append(point22)
    if len(pointsL) == 0 and len(pointsR) == 0:
        return None, None, None
    return pointsL, pointsR, RR_start_points


#########################

def generate_epipolar_points(dataL, dataR, rpc1, rpc2, h_Max=1000, h_Min=0):
    # set the max and min height
    # h_Max = 1500
    # h_Min = 0
    # h_constant = (h_Max+h_Min)/2
    h_constant = 0

    Max_y1 = dataL.shape[0]
    Max_x1 = dataL.shape[1]
    Max_y2 = dataR.shape[0]
    Max_x2 = dataR.shape[1]

    L_points = []
    R_points = []
    R_start_points = []
    R_start_point = point()
    initial = point()

    initial.x = (Max_x1 / 2)
    initial.y = (Max_y1 / 2)
    # RPC progress
    L_points.append(initial)
    R_start_point = get_Rstartpoint(rpc1, rpc2, initial, h_constant)
    R_start_points.append(R_start_point)
    point1, point2 = Left2Right1(rpc1, rpc2, initial, h_Min, h_Max)
    R_points.append(point1)
    R_points.append(point2)
    point1, point2 = Right2Left(rpc1, rpc2, point1, point2, h_Min, h_Max)
    L_points.append(point1)
    L_points.append(point2)
    while 0 < point1.x < Max_x1 and 0 < point1.y < Max_y1 and 0 < point2.x < Max_x2 and 0 < point2.y < Max_y2:
        point1, point2 = Left2Right2(rpc1, rpc2, point1, point2, h_Min, h_Max)
        R_points.append(point1)
        R_points.append(point2)
        point1, point2 = Right2Left(rpc1, rpc2, point1, point2, h_Min, h_Max)
        L_points.append(point1)
        L_points.append(point2)

    L_points = Basefunction.sort_points(L_points)
    number = int(len(L_points))
    inter_dis = math.sqrt(
        (L_points[0].x - L_points[number - 1].x) ** 2 + (L_points[0].y - L_points[number - 1].y) ** 2) / (number - 1)
    dis = math.sqrt(Max_x1 * Max_x1 + Max_y1 * Max_y1)
    num = int(dis / 2 / inter_dis) + 1
    LL_coef = []
    RR_coef = []
    L_coe = Basefunction.eastsq_fit(L_points)
    R_coe = Basefunction.eastsq_fit(R_points)
    LL_coef.append(L_coe)
    RR_coef.append(R_coe)
    k, b = Basefunction.get_perpendicular(L_coe[0][0], L_coe[0][1], initial)
    k2, b2 = Basefunction.get_perpendicular(R_coe[0][0], R_coe[0][1], R_start_points[0])
    pointsss = Basefunction.get_area(k, b, dataL, inter_dis)

    # print("first Epipolar line is Ok")

    # L_points = Basefunction.sort_points(L_points)
    RR_points = []
    LL_points = []
    RR_startpoints = []
    LL_points_all = []
    RR_points_all = []
    RR_startpoints_all = []
    for i in range(len(pointsss)):
        L, R, RS = RPC_process(rpc1, rpc2, pointsss[i].x, pointsss[i].y, h_Min, h_Max, h_constant, dataL, dataR, num)
        if L is None or R is None or RS is None:
            continue
        RR_points.append(R)
        LL_points.append(L)
        RR_startpoints.append(RS)

    p = int((len(pointsss) - len(LL_points)) / 2)
    for i in range(len(pointsss)):
        L, R, RS = RPC_process2(rpc1, rpc2, pointsss[i].x, pointsss[i].y, h_Min, h_Max, h_constant, dataL, dataR, num)
        if L is None or R is None or RS is None:
            continue
        RR_points_all.append(R)
        LL_points_all.append(L)
        RR_startpoints_all.append(RS)

    return LL_points, LL_points_all, RR_points, RR_points_all, initial, R_start_point, L_coe, R_coe
