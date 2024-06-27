import os
import re
import numpy as np
import string
from osgeo import gdal
import time
from Basefunction import point
import matplotlib .pyplot as plt
from filesave import Save_as_tif
import cv2
from GDALfilestest import arr2raster



class RPCL():
    '''
    The calss of RPC pars of Left image.
    '''
    lineNumCoef = []
    lineDenCoef = []
    sampNumCoef = []
    sampDenCoef = []
    errBias = 0
    errRand = 0
    lineOffset = 0
    sampOffset = 0
    latOffset = 0
    longOffset = 0
    heightOffset = 0
    lineScale = 0
    sampScale = 0
    latScale = 0
    longScale = 0
    heightScale = 0

class RPCR():
    '''
    The class of RPC pars of Right image
    '''
    lineNumCoef = []
    lineDenCoef = []
    sampNumCoef = []
    sampDenCoef = []
    errBias = 0
    errRand = 0
    lineOffset = 0
    sampOffset = 0
    latOffset = 0
    longOffset = 0
    heightOffset = 0
    lineScale = 0
    sampScale = 0
    latScale = 0
    longScale = 0
    heightScale = 0


def pre_process_RPCL(rpc):
    '''
    the process of extrating pars of rpc to Fixed data structure
    :param rpc: rpc read from rpc files
    :return: fixed rpc data structure
    '''
    samplel = RPCL()
    # samplel.errBias = rpc['ERR_BIAS']
    # samplel.errRand = rpc['ERR_RAND']
    samplel.lineOffset = float(rpc['LINE_OFF'].replace(" pixels", ""))
    samplel.sampOffset = float(rpc['SAMP_OFF'].replace(" pixels", ""))
    samplel.latOffset = float(rpc['LAT_OFF'].replace(" degrees", ""))
    samplel.longOffset = float(rpc['LONG_OFF'].replace(" degrees", ""))
    samplel.heightOffset = float(rpc['HEIGHT_OFF'].replace(" meters", ""))
    samplel.lineScale = float(rpc['LINE_SCALE'].replace(" pixels", ""))
    samplel.sampScale = float(rpc['SAMP_SCALE'].replace(" pixels", ""))
    samplel.latScale = float(rpc['LAT_SCALE'].replace(" degrees", ""))
    samplel.longScale = float(rpc['LONG_SCALE'].replace(" degrees", ""))
    samplel.heightScale = float(rpc['HEIGHT_SCALE'].replace(" meters", ""))

    string1 = rpc['SAMP_DEN_COEFF'].split()
    string2 = rpc['SAMP_NUM_COEFF'].split()
    string3 = rpc['LINE_DEN_COEFF'].split()
    string4 = rpc['LINE_NUM_COEFF'].split()

    for i in range(20):
        samplel.sampDenCoef.append(float(string1[i]))
        samplel.sampNumCoef.append(float(string2[i]))
        samplel.lineDenCoef.append(float(string3[i]))
        samplel.lineNumCoef.append(float(string4[i]))

    return samplel


def pre_process_RPCR(rpc):
    sampler = RPCR()
    # sampler.errBias = rpc['ERR_BIAS']
    # sampler.errRand = rpc['ERR_RAND']
    sampler.lineOffset = float(rpc['LINE_OFF'].replace(" pixels", ""))
    sampler.sampOffset = float(rpc['SAMP_OFF'].replace(" pixels", ""))
    sampler.latOffset = float(rpc['LAT_OFF'].replace(" degrees", ""))
    sampler.longOffset = float(rpc['LONG_OFF'].replace(" degrees", ""))
    sampler.heightOffset = float(rpc['HEIGHT_OFF'].replace(" meters", ""))
    sampler.lineScale = float(rpc['LINE_SCALE'].replace(" pixels", ""))
    sampler.sampScale = float(rpc['SAMP_SCALE'].replace(" pixels", ""))
    sampler.latScale = float(rpc['LAT_SCALE'].replace(" degrees", ""))
    sampler.longScale = float(rpc['LONG_SCALE'].replace(" degrees", ""))
    sampler.heightScale = float(rpc['HEIGHT_SCALE'].replace(" meters", ""))

    string1 = rpc['SAMP_DEN_COEFF'].split()
    string2 = rpc['SAMP_NUM_COEFF'].split()
    string3 = rpc['LINE_DEN_COEFF'].split()
    string4 = rpc['LINE_NUM_COEFF'].split()

    for i in range(20):
        sampler.sampDenCoef.append(float(string1[i]))
        sampler.sampNumCoef.append(float(string2[i]))
        sampler.lineDenCoef.append(float(string3[i]))
        sampler.lineNumCoef.append(float(string4[i]))

    return sampler


def cal_Ploy3(coef, U, V, W):
    '''
    Calculate third order polynomial
    :param coef: coefficient of polynomial
    :param U:
    :param V:
    :param W:
    :return:
    '''
    u = np.mat([1, V, U, W, V*U, V*W, U*W, V*V, U*U, W*W, U*V*W, V*V*V, V*U*U, V*W*W, V*V*U, U*U*U, U*W*W, V*V*W, U*U*W, W*W*W]).T
    coeff = np.mat(coef)
    result = np.dot(coeff, u)
    return float(result)

def cal_rfm(Num, Den, x, y, z):
    m = cal_Ploy3(Num, x, y, z)
    n = cal_Ploy3(Den, x, y, z)
    result = m / n
    return result

# Get l and s by RPC model
def RPC_Positive(long, lat, height, rpc):
    '''
    RPC_positive: lon, lat and height ---> l and s
    :param long: longtitude
    :param lat: latitude
    :param height:
    :param rpc:
    :return:
    '''
    U = (lat - rpc.latOffset) / rpc.latScale
    V = (long - rpc.longOffset) / rpc.longScale
    W = (height - rpc.heightOffset) / rpc.heightScale

    X = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, U, V, W)
    Y = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, U, V, W)
    
    s = rpc.sampOffset + rpc.sampScale * X
    l = rpc.lineOffset + rpc.lineScale * Y

    return l, s

# line , sample and hegiht --> long and lat  by RFM
def RPC_Negative_inverse(samp, line , height, rpc):

    Xf = np.vstack([line, samp]).T

    long = -line ** 0
    lat = -samp ** 0

    EPS = 2
    x0 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat, long, height)
    y0 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat, long, height)
    x1 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat, long + EPS, height)
    y1 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat, long + EPS, height)
    x2 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat + EPS, long, height)
    y2 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat + EPS, long, height)

    n = 0
    while not np.all((x0 - line) ** 2 + (y0 - samp) ** 2 < 1e-20):

        if n > 1000:
            print("error,exceed 1000 times RPC_Negative")
            pass
        X0 = np.vstack([x0, y0]).T
        X1 = np.vstack([x1, y1]).T
        X2 = np.vstack([x2, y2]).T
        e1 = X1 - X0
        e2 = X2 - X0
        u = Xf - X0

        num = np.sum(np.multiply(u, e1), axis=1)
        den = np.sum(np.multiply(e1, e1), axis=1)
        a1 = np.divide(num, den).squeeze()
        num = np.sum(np.multiply(u, e2), axis=1)
        den = np.sum(np.multiply(e2, e2), axis=1)
        a2 = np.divide(num, den).squeeze()

        long = long + a1 * EPS
        lat = lat + a2 * EPS

        EPS = .1
        x0 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat, long, height)
        y0 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat, long, height)
        x1 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat, long + EPS, height)
        y1 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat, long + EPS, height)
        x2 = cal_rfm(rpc.lineNumCoef, rpc.lineDenCoef, lat + EPS, long, height)
        y2 = cal_rfm(rpc.sampNumCoef, rpc.sampDenCoef, lat + EPS, long, height)

        n += 1
    return long, lat


def RPC_Negative(line, samp, height, rpc):
    line = (line - rpc.lineOffset) / rpc.lineScale
    samp = (samp - rpc.sampOffset) / rpc.sampScale
    height = (height - rpc.heightOffset) / rpc.heightScale

    long, lat = RPC_Negative_inverse(samp, line, height, rpc)

    long = long * rpc.longScale + rpc.longOffset
    lat = lat * rpc.latScale + rpc.latOffset

    return float(long), float(lat)

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
    lon, lat = RPC_Negative(initial_point.y, initial_point.x, h, rpc1)
    temp_point.y, temp_point.x = RPC_Positive(lon, lat, h, rpc2)
    return temp_point

if __name__ == "__main__":
    Left = "K:\ZY3_inhanced\ZY3_01a_hsnbavp_897148_20130812_111815_0008_SASMAC_CHN_sec_rel_001_1308281144.tif"
    Right = "K:\ZY3_inhanced\ZY3_01a_hsnfavp_897148_20130812_111718_0008_SASMAC_CHN_sec_rel_001_1308281300.tif"
    # Left = "K:\ZY3_SPRS\ZY3_01a_mynbavp_278116_20140827_183905_0008_SASMAC_CHN_sec_rel_001_14082905653\ZY3_01a_mynbavp_278116_20140827_183905_0008_SASMAC_CHN_sec_rel_001_14082905653\ZY3_01a_mynbavp_278116_20140827_183905_0008_SASMAC_CHN_sec_rel_001_14082905653.tif"
    # Right = "K:\ZY3_SPRS\ZY3_01a_mynfavp_278116_20140827_183807_0008_SASMAC_CHN_sec_rel_001_14082905709\ZY3_01a_mynfavp_278116_20140827_183807_0008_SASMAC_CHN_sec_rel_001_14082905709\ZY3_01a_mynfavp_278116_20140827_183807_0008_SASMAC_CHN_sec_rel_001_14082905709.tif"
    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
    # dataL = datasetL.ReadAsArray(0, 0, 5000, 5000)
    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)
    # dataR = datasetR.ReadAsArray(2, 0, 5000, 5000)
    dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize - 0)
    tempL = datasetL.GetMetadata("RPC")
    tempR = datasetR.GetMetadata("RPC")
    rpc1 = pre_process_RPCL(tempL)
    rpc2 = pre_process_RPCR(tempR)
    print("Successfully read Images")
    # set the max and min height
    h_Max = 1500
    h_Min = 0
    # h_constant = (h_Max+h_Min)/2
    h_constant = 0

    Max_y1 = dataL.shape[0]
    Max_x1 = dataL.shape[1]
    Max_y2 = dataR.shape[0]
    Max_x2 = dataR.shape[1]
    # Max_y1 = 5000
    # Max_x1 = 5000
    # Max_y2 = 5000
    # Max_x2 = 5000

    L_points = []
    R_points = []
    R_start_points = []

    initial = point()
    R_start_point = point()
    initial.x = Max_x1 / 2
    initial.y = Max_y1 / 2
    R_cenP = point()
    R_cenP.x = Max_x2 / 2
    R_cenP.y = Max_y2 / 2

    L_points.append(initial)
    R_start_point = get_Rstartpoint(rpc1, rpc2, initial, 0)

    temp = get_Rstartpoint(rpc2, rpc1, R_start_point, 0)
    print("Xdis", initial.x - R_start_point.x)
    print("Ydis", initial.y - R_start_point.y)