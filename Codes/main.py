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
import matplotlib .pyplot as plt
import time
import Polytran_and_resample
from filesave import Save_as_tif
from Polytran_and_resample import cal_chunk
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
    lon1, lat1 = rpc.RPC_Negative( initial.y, initial.x, h_Min, rpc1)
    lon2, lat2 = rpc.RPC_Negative( initial.y, initial.x, h_Max, rpc1)
    temp_point1.y, temp_point1.x = rpc.RPC_Positive(lon1, lat1, h_Min, rpc2)
    temp_point2.y, temp_point2.x = rpc.RPC_Positive(lon2, lat2, h_Max,rpc2)
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
    temp_point1.y, temp_point1.x = rpc.RPC_Positive( lon1, lat1, h_Max, rpc1)
    temp_point2.y, temp_point2.x = rpc.RPC_Positive( lon2, lat2, h_Min, rpc1)
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
    lon1, lat1 = rpc.RPC_Negative( point1.y, point1.x, h_Min, rpc1)
    lon2, lat2 = rpc.RPC_Negative(point2.y, point2.x, h_Max, rpc1)
    temp_point1.y, temp_point1.x = rpc.RPC_Positive( lon1, lat1, h_Min, rpc2)
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

def RPC_process(rpc1, rpc2, x, y, h_Min, h_Max, h_constant, dataL, dataR, num):
    '''

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
    if(cal_position(ini_point, dataL)):
        pointsL.append(ini_point)
    RR_start_point = get_Rstartpoint(rpc1, rpc2, ini_point, h_constant)
    if(cal_position(RR_start_point, dataR)):
        RR_start_points.append(RR_start_point)
    point11, point22 = Left2Right1(rpc1, rpc2, ini_point, h_Min, h_Max)
    # if (cal_position(point11, dataR) == 0 or cal_position(point22, dataR) == 0):
    #     return None, None, None
    # else:
    #     pointsR.append(point11)
    #     pointsR.append(point22)
    if (cal_position(point11, dataR)):
        pointsR.append(point11)
    if (cal_position(point22, dataR)):
        pointsR.append(point22)


    point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
    # if (cal_position(point11, dataL) == 0 and cal_position(point22, dataL) == 0):
    #     return pointsL, pointsR, RR_start_points
    # else:
    if (cal_position(point11, dataL)):
        pointsL.append(point11)
    if (cal_position(point22, dataL)):
        pointsL.append(point22)

    # while ( cal_position(point11, dataL) or cal_position(point22, dataL)):
    # a, b = point11, point22
    # while (cal_position(point11, dataL) or cal_position(point22, dataL) or cal_position(a, dataR) or cal_position(b, dataR)):
    for i in range(num):
        point11, point22 = Left2Right2(rpc1, rpc2, point11, point22, h_Min, h_Max)
        if (cal_position(point11, dataR)):
            pointsR.append(point11)
        if (cal_position(point22, dataR)):
            pointsR.append(point22)
        point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
        if (cal_position(point11, dataL)):
            pointsL.append(point11)
        if (cal_position(point22, dataL)):
            pointsL.append(point22)
    if (len(pointsL)== 0 or len(pointsR) == 0):
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
    if (len(pointsL)== 0 and len(pointsR) == 0):
        return None, None, None
    return pointsL, pointsR, RR_start_points
#########################

if __name__ == "__main__":
    t1 = time.time()
    Left = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff"
    Right = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-FWDPAN.tiff"
    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)
    dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)
    tempL = datasetL.GetMetadata("RPC")
    tempR = datasetR.GetMetadata("RPC")
    rpc1 = rpc.pre_process_RPCL(tempL)
    rpc2 = rpc.pre_process_RPCR(tempR)
    t2 = time.time()
    print("Successfully read Images. Time:", t2-t1, "s")

    # set the max and min height
    h_Max = 2500
    h_Min = 0
    # h_constant = (h_Max+h_Min)/2
    h_constant = 0

    Max_y1 = datasetL.RasterYSize
    Max_x1 = datasetL.RasterXSize
    Max_y2 = datasetR.RasterYSize
    Max_x2 = datasetR.RasterXSize

    L_points = []
    R_points = []
    R_start_points = []

    initial = point()
    R_start_point = point()
    initial.x = int(Max_x1 / 2)
    initial.y = int(Max_y1 / 2)
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
    while (
            point1.x > 0 and point1.x < Max_x1 and point1.y > 0 and point1.y < Max_y1 and point2.x > 0 and point2.x < Max_x2 and point2.y > 0 and point2.y < Max_y2):
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

    t1 = time.time()
    L_points = Basefunction.sort_points(L_points)
    RR_points = []
    LL_points = []
    RR_startpoints = []
    LL_points_all = []
    RR_points_all = []
    RR_startpoints_all = []
    for i in range(len(pointsss)):
        L , R, RS= RPC_process(rpc1, rpc2, pointsss[i].x, pointsss[i].y, h_Min, h_Max, h_constant, dataL, dataR, num)
        if ( L == None or R == None or RS == None):
            continue
        RR_points.append(R)
        LL_points.append(L)
        RR_startpoints.append(RS)

    p = int((len(pointsss) - len(LL_points))/2)
    for i in range(len(pointsss)):
        L, R, RS = RPC_process2(rpc1, rpc2, pointsss[i].x, pointsss[i].y, h_Min, h_Max, h_constant, dataL, dataR, num)
        if (L == None or R == None or RS == None):
            continue

        RR_points_all.append(R)
        LL_points_all.append(L)
        RR_startpoints_all.append(RS)

    t2 = time.time()
    # print("Epipolar Lines are calcualted! ")
    print("epipolar points generation time:", t2-t1, "s")


    t1 = time.time()
    mmm, nnn = relocate.relocate_sub2(initial, LL_points_all, L_coe[0][0], R_start_point, RR_points_all, R_coe[0][0])
    t2 = time.time()
    # print("Rolate is OK!")
    print("epipolar points rolation time:", t2 - t1, "s")

    t1 = time.time()
    L_grids = Polytran_and_resample.Normalize_grids(LL_points_all, mmm)
    R_grids = Polytran_and_resample.Normalize_grids(RR_points_all, nnn)
    # print("Grids calculated OK!")
    mmm_in = Polytran_and_resample.sift_grids(L_grids, dataL)
    nnn_in = Polytran_and_resample.sift_grids(R_grids, dataR)


    poly_L, poly_R, x_cof, y_range = Polytran_and_resample.Determin_region(mmm_in, nnn_in)
    # L_grids = L_grids[0:10]

    L_tans = Cal_epipolar_geotrans_L(datasetL, L_grids, x_cof, y_range)
    R_tans = Cal_epipolar_geotrans_R(datasetR, R_grids, x_cof, y_range)
    t2 = time.time()
    print("Calc Aff_coff time:", t2 - t1, "s")
    dataL = 0
    datasetL = 0
    datasetR = 0

    # newLL = np.zeros((60000, 60000), dtype=np.int16)
    #
    # # chunk_L = cal_chunk(L_grids, dataL)
    # startt = time.time()
    # chunk_L = cal_chunk(L_grids, dataL)
    # endt = time.time()
    # print("L_img Interpolation time:", endt-startt, "s")
    #
    # t1 = time.time()
    # newLL = Polytran_and_resample.value_chunk(chunk_L, newLL, L_grids)
    # newLL = Polytran_and_resample.cut_minpolygon_L(newLL, x_cof, y_range)
    # Save_as_tif(Left,"K:\data\GF\L_img.tiff",newLL, band=1)
    # newLL = 0
    # t2 = time.time()
    # print("left epipolar image is saved! time:", t2 - t1, "s")
    newRR = np.zeros((60000, 60000), dtype=np.int16)
    t1 = time.time()
    chunk_R = cal_chunk(R_grids, dataR)
    t2 = time.time()
    print("R_img Interpolation time:", t2 - t1, "s")

    t1 = time.time()
    newRR = Polytran_and_resample.value_chunk(chunk_R, newRR, R_grids)
    newRR = Polytran_and_resample.cut_minpolygon_R(newRR, x_cof, y_range)
    Save_as_tif(Right, "K:\data\GF\R_img.tiff", newRR, band=1)
    t2 = time.time()
    newRR = 0
    print("right epipolar image is saved! time:", t2 - t1, "s")








