import numpy as np
import os
import rpc
from osgeo import gdal
from Basefunction import point
from Basefunction import cal_coff_aff
from Basefunction import cal_position
import Basefunction
import relocate
import math
import matplotlib .pyplot as plt
import time
import Polytran_and_resample
# line is y, samp is x

# Step1 one point from LeftImage to rightImage
def Left2Right1(rpc1, rpc2, x, y, h_Min, h_Max):
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
    lon1, lat1 = rpc.RPC_Negative( x, y, h_Min,rpc1)
    lon2,  lat2 = rpc.RPC_Negative( x, y, h_Max, rpc1)
    temp_point1.x, temp_point1.y = rpc.RPC_Positive(lon1, lat1, h_Min, rpc2)
    temp_point2.x, temp_point2.y = rpc.RPC_Positive( lon2, lat2, h_Max,rpc2)
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
    lon1, lat1 = rpc.RPC_Negative(point1.x, point1.y, h_Max, rpc2)
    lon2, lat2 = rpc.RPC_Negative(point2.x, point2.y, h_Min, rpc2)
    temp_point1.x, temp_point1.y = rpc.RPC_Positive( lon1, lat1, h_Max, rpc1)
    temp_point2.x, temp_point2.y = rpc.RPC_Positive(lon2, lat2, h_Min, rpc1)
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
    lon1, lat1 = rpc.RPC_Negative( point1.x, point1.y, h_Min, rpc1)
    lon2, lat2 = rpc.RPC_Negative(point2.x, point2.y, h_Max, rpc1)
    temp_point1.x, temp_point1.y = rpc.RPC_Positive( lon1, lat1, h_Min, rpc2)
    temp_point2.x, temp_point2.y = rpc.RPC_Positive(lon2, lat2, h_Max, rpc2)
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
    lon, lat = rpc.RPC_Negative(initial_point.x, initial_point.y, h, rpc1)
    temp_point.x, temp_point.y = rpc.RPC_Positive(lon, lat, h, rpc2)
    return temp_point

def RPC_process(rpc1, rpc2, x, y, h_Min, h_Max, h_constant, dataL, dataR):
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
    pointsL.append(ini_point)
    RR_start_point = get_Rstartpoint(rpc1, rpc2, ini_point, h_constant)
    RR_start_points.append(RR_start_point)
    point11, point22 = Left2Right1(rpc1, rpc2, ini_point.x, ini_point.y, h_Min, h_Max)
    if (cal_position(point11, dataR) == 0 or cal_position(point22, dataR) == 0):
        return None, None, None
    else:
        pointsR.append(point11)
        pointsR.append(point22)


    point11, point22 = Right2Left(rpc1, rpc2, point11, point22, h_Min, h_Max)
    if (cal_position(point11, dataL) == 0 or cal_position(point22, dataL) == 0):
        return None, None, None
    else:
        pointsL.append(point11)
        pointsL.append(point22)
    while ( cal_position(point11, dataL) or cal_position(point22, dataL)):
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

    return pointsL, pointsR, RR_start_points

#########################
# read the rpc files
Left = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff"
Right = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-FWDPAN.tiff"

# Left = "K:\BaiduNetdiskDownload\ZY3_Wuhan\ZY3_Wuhan\ZY3_01a_hsnbavp_897148_20130812_111815_0008_SASMAC_CHN_sec_rel_001_1308281144\ZY3_01a_hsnbavp_897148_20130812_111815_0008_SASMAC_CHN_sec_rel_001_1308281144.tif"
# Right = "K:\BaiduNetdiskDownload\ZY3_Wuhan\ZY3_Wuhan\ZY3_01a_hsnfavp_897148_20130812_111718_0008_SASMAC_CHN_sec_rel_001_1308281300\ZY3_01a_hsnfavp_897148_20130812_111718_0008_SASMAC_CHN_sec_rel_001_1308281300.tif"
datasetL = gdal.Open(Left, gdal.GA_Update)
dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
datasetR = gdal.Open(Right, gdal.GA_Update)
dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)
tempL = datasetL.GetMetadata("RPC")
tempR = datasetR.GetMetadata("RPC")
rpc1 = rpc.pre_process_RPCL(tempL)
rpc2 = rpc.pre_process_RPCR(tempR)
print("Successfully read Images")


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
initial.x = int(Max_x1/2)
initial.y = int(Max_y1/2)
# RPC progress
L_points.append(initial)
R_start_point = get_Rstartpoint(rpc1, rpc2, initial, h_constant)
R_start_points.append(R_start_point)
point1, point2 = Left2Right1(rpc1, rpc2, initial.x, initial.y, h_Min, h_Max)
R_points.append(point1)
R_points.append(point2)
point1, point2 = Right2Left(rpc1, rpc2, point1, point2, h_Min, h_Max)
L_points.append(point1)
L_points.append(point2)
while(point1.x > 0 and point1.x<Max_x1 and point1.y>0 and point1.y<Max_y1 and point2.x>0 and point2.x<Max_x2 and point2.y>0 and point2.y<Max_y2):
    point1, point2 = Left2Right2(rpc1, rpc2, point1, point2, h_Min, h_Max)
    R_points.append(point1)
    R_points.append(point2)
    point1, point2 = Right2Left(rpc1, rpc2, point1, point2, h_Min, h_Max)
    L_points.append(point1)
    L_points.append(point2)
LL_coef = []
RR_coef = []
L_coe = Basefunction.eastsq_fit(L_points)
R_coe = Basefunction.eastsq_fit(R_points)
LL_coef.append(L_coe)
RR_coef.append(R_coe)
k, b = Basefunction.get_perpendicular(L_coe[0][0], L_coe[0][1], initial)
k2, b2 = Basefunction.get_perpendicular(R_coe[0][0], R_coe[0][1], R_start_points[0])
pointsss = Basefunction.get_area(k, b, datasetL)
pointssss = Basefunction.get_area(k2, b2, datasetR)
print("first Epipolar line is Ok")

# testx, testy = Basefunction.extra_xy(R_points)
# testxx, testyy = Basefunction.extra_xy(pointssss)
# length = len(pointssss)
# # plt.plot(testx, testy,"o", color='red', markersize=2)
# plt.plot([1,Max_x2-1],[R_coe[0][0] + R_coe[0][1],R_coe[0][0]*(Max_x1-1)+R_coe[0][1]], 'r-', linewidth=2)
# plt.plot([pointssss[0].x, pointssss[length-1].x], [pointssss[0].y, pointssss[length-1].y], 'r-', linewidth=2)

# plt.plot(testxx, testyy,"o", color="red", markersize=2)
# # plt.plot(initial.x, initial.y, "o", color="red", markersize=2)
# plt.imshow(dataL, cmap='gray')
# plt.show()
# print("OK")

###############################




if __name__ == "__main__":
    RR_points = []
    LL_points = []
    RR_startpoints = []
    for i in range(len(pointsss)):
        L , R, RS= RPC_process(rpc1, rpc2, pointsss[i].x, pointsss[i].y, h_Min, h_Max, h_constant, dataL, dataR)
        if ( L == None or R == None or RS == None):
            continue
        L_coef = Basefunction.eastsq_fit(L)
        R_coef = Basefunction.eastsq_fit(R)
        LL_coef.append(L_coef)
        RR_coef.append(R_coef)
        RR_points.append(R)
        LL_points.append(L)
        RR_startpoints.append(RS)
    # L_poly = Polytran_and_resample.Cal_Minpolygon(LL_points)
    # R_poly = Polytran_and_resample.Cal_Minpolygon(RR_points)

    # testx, testy = Basefunction.extra_xys(RR_points)
    # minx, maxx = min(testx), max(testx)
    # miny, maxy = min(testy), max(testy)
    # print("minx and maxx are ", int(minx), int(maxx))
    # print("miny and maxy are ", int(miny), int(maxy))
    # dataR = dataR.astype(np.int16)
    # # plt.imshow(dataL)
    # plt.plot(testx, testy, "o", color="red", markersize=2)
    # plt.imshow(dataR, cmap="gray")
    # # plt.show()
    # # for i in range(len(LL_coef)):
    # #     plt.plot([1, Max_x1 - 1],
    # #              [LL_coef[i][0][0] - LL_coef[i][0][1], LL_coef[i][0][0] * (Max_x1 - 1) - LL_coef[i][0][1]], 'r-',
    # #              linewidth=2)
    # plt.show()

    print("Epipolar Lines are calcualted! ")

    # revolve and relate
    angleL = math.atan(L_coe[0][0])
    mm = relocate.lines_rotate(angleL, initial, LL_points)
    angleR = math.atan(R_coe[0][0])
    nn = relocate.lines_rotate(angleR, R_start_point, RR_points )
    print("Revolve is OK!")
    mmm, nnn = relocate.change_y_lines(LL_points, RR_points)
    print("Rolate is OK!")

    # affiane transform
    L_aff_original_points = []
    L_aff_temp_points = []
    R_aff_original_points = []
    R_aff_temp_points = []
    for i in range(len(LL_points)):
        for j in range(len(LL_points[i])):
            L_aff_original_points.append(LL_points[i][j])
            L_aff_temp_points.append(mmm[i][j])
    for i in range(len(RR_points)):
        for j in range(len(RR_points[i])):
            R_aff_original_points.append((RR_points[i][j]))
            R_aff_temp_points.append(nnn[i][j])

    cof_aff_L = cal_coff_aff(L_aff_original_points,L_aff_temp_points)
    cof_aff_R = cal_coff_aff(R_aff_original_points,R_aff_temp_points)

    cof_aff_L_inverse = cal_coff_aff(L_aff_temp_points, L_aff_original_points)
    cof_aff_R_inverse = cal_coff_aff(R_aff_temp_points, R_aff_original_points)
    print("image aff cof is calculated! ")

    # # test the cof
    # testpoint= Basefunction.get_changed_xy(R_aff_original_points[0].x, R_aff_original_points[0].y, cof_aff_R)
    # print("original point x y",R_aff_original_points[0].x, R_aff_original_points[0].y )
    # print(" changed point x y",R_aff_temp_points[0].x, R_aff_temp_points[0].y)
    # print("cal point x y", testpoint.x, testpoint.y)

    # Resampling images using bilinear interpolation
    min_x, max_x, min_y, max_y = Basefunction.cal_region(dataL, cof_aff_L)
    min_x2, max_x2, min_y2, max_y2 = Basefunction.cal_region(dataR, cof_aff_R)

    mm_minx = min(min_x, min_x2)
    mm_miny = min(min_y, min_y2)
    # cal the step of translation
    if (mm_minx > 0):
        step_x = int(np.ceil(mm_minx))
    elif(mm_minx == 0):
        step_x = int(0)
    else:
        step_x = int(np.floor(mm_minx))

    if (mm_miny > 0):
        step_y = int(np.ceil(mm_miny))
    elif(mm_miny == 0):
        step_y = int(0)
    else:
        step_y = int(np.floor(mm_miny))

    # 确定新影像的范围

    distance_x =max(max_x, max_x2) - min(min_x, min_x2)
    distance_y = max(max_y, max_y2) - min(min_y, min_y2)


    Start_resample_time = time.time()
    newR = np.zeros((int(distance_x+1), int(distance_y+1)),dtype=np.int16)
    # newR = np.zeros((int(distance_x + 1), int(distance_y + 1)), dtype=np.int16)
    cof = cof_aff_R_inverse

    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = cof[0, 0], cof[1, 0], cof[2, 0], cof[3, 0], cof[4, 0], cof[5, 0], cof[6, 0], cof[7, 0], cof[8, 0], cof[9, 0]
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 = cof[10, 0], cof[11, 0], cof[12, 0], cof[13, 0], cof[14, 0], cof[15, 0], cof[16, 0], cof[17, 0], cof[18, 0], cof[19, 0]

    for i in range( int(newR.shape[0])):
        if (i%10 == 0):
            print("R num ",i, "is OK")
        for j in range(int(newR.shape[1])):
            # temppt = Basefunction.get_changed_xy(i, j, cof_aff_L_inverse)
            # newL[i-step_x][j-step_y]  = Basefunction.bilinear_interpolation(temppt, dataL)
            temptt = Basefunction.get_changed_xy_fast(i+step_x, j+step_y,a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,  b1, b2, b3, b4, b5, b6, b7, b8, b9, b10)
            newR[i][j] = Basefunction.bilinear_interpolation(temptt, dataR)

    End_resample_time = time.time()
    Runtime = End_resample_time - Start_resample_time
    print("Resampling R_image_points time is ", Runtime," sec")
    orginal_file = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-FWDPAN.tiff"
    driver = gdal.GetDriverByName("GTiff")
    etype = gdal.GDT_Int16
    proj = gdal.Open(orginal_file).GetProjection()
    transform = gdal.Open(orginal_file).GetGeoTransform()
    result_file = "K:/data/R_curve_all.tiff"
    ds = driver.Create(result_file, newR.shape[1], newR.shape[0], 1, etype)  # 行，列
    ds.GetRasterBand(1).WriteArray(newR)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds

    Start_resample_time2 = time.time()
    newL = np.zeros((int(distance_x + 1), int(distance_y + 1)), dtype=np.int16)
    # newR = np.zeros((int(distance_x + 1), int(distance_y + 1)), dtype=np.int16)
    cof = cof_aff_L_inverse

    a1, a2, a3, a4, a5, a6 = cof[0, 0], cof[1, 0], cof[2, 0], cof[3, 0], cof[4, 0], cof[5, 0]
    b1, b2, b3, b4, b5, b6 = cof[6, 0], cof[7, 0], cof[8, 0], cof[9, 0], cof[10, 0], cof[11, 0]

    for i in range(0, int(newL.shape[0] )):
        if (i % 10 == 0):
            print("L num ", i, "is OK")
        for j in range(0, int(newL.shape[1] )):
            # temppt = Basefunction.get_changed_xy(i, j, cof_aff_L_inverse)
            # newL[i-step_x][j-step_y]  = Basefunction.bilinear_interpolation(temppt, dataL)
            temptt = Basefunction.get_changed_xy_fast(i + step_x, j + step_y, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4,
                                                      b5, b6)
            newL[i][j] = Basefunction.bilinear_interpolation(temptt, dataL)

    End_resample_time2 = time.time()
    Runtime = End_resample_time2 - Start_resample_time2
    print("Resampling L_image points time is ", Runtime, " sec")

    orginal_file = "K:\data\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff"
    driver = gdal.GetDriverByName("GTiff")
    etype = gdal.GDT_Int16
    proj = gdal.Open(orginal_file).GetProjection()
    transform = gdal.Open(orginal_file).GetGeoTransform()
    result_file = "K:/data/L_curve_all.tiff"
    ds = driver.Create(result_file, newL.shape[1], newL.shape[0], 1, etype)  # 行，列
    ds.GetRasterBand(1).WriteArray(newL)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds


    print("OK")


