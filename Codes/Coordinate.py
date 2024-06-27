import numpy as np
import Basefunction
from Basefunction import point
from Basefunction import Deterimin_in_polygon
from Basefunction import Deterimin_in_polygon2
from osgeo import gdal
from Basefunction import Aff_xy

def rowcol2geo(geoTransform, row, col):
    '''
    利用GDAL的仿射变换矩阵把行列号转换为地理坐标
    :param geoTransform:
    :param row:
    :param col:
    :return:
    '''
    pointt = point()
    pointt.x = geoTransform[0] + col * geoTransform[1] + row * geoTransform[2]
    pointt.y = geoTransform[3] + col * geoTransform[4] + row * geoTransform[5]
    return pointt

def Cal_cornerpoint_geo_cor(trans, img):
    '''
    计算影像四个角点的地理坐标，通过GDAL的仿射变换参数
    :param trans:
    :param img:
    :return:
    '''
    pointA = rowcol2geo(trans,0, 0)
    pointB = rowcol2geo(trans, 0, img.shape[1])
    pointC = rowcol2geo(trans, img.shape[0], 0)
    pointD = rowcol2geo(trans, img.shape[0], img.shape[1])
    points = []
    points.append(pointA)
    points.append(pointB)
    points.append(pointC)
    points.append(pointD)
    return pointA, pointB, pointC, pointD

def Cal_trans_fromCGPS(Grids, points):
    points_epipolar = []
    for i in range(len(points)):
        for j in range(len(Grids)):
            pointp = points[i]
            pointA = Grids[j].grid_old.point1
            pointB = Grids[j].grid_old.point2
            pointC = Grids[j].grid_old.point4
            pointD = Grids[j].grid_old.point3
            if (Deterimin_in_polygon(pointA, pointB, pointC, pointD, pointp) == 1):
                a1, a2, a3, b1, b2, b3 = Grids[j].cof[0:6]
                a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)
                tp = point()
                tp.y = (a2*points[i].y - a2 * b1 - b2 * points[i].x + a1 * b2) / (a2*b3 - a3*b2)
                tp.x = (points[i].x - a1 - a3 * tp.y)/a2
                points_epipolar.append(tp)
    return points_epipolar

def calc_cut_epointsL(points, x_cof, y_range):
    '''
    计算切割后的核线影像坐标
    :param points:
    :param x_cof:
    :param y_range:
    :return:
    '''
    pointss = []
    for i in range(len(points)):
        tp = point()
        tp.x = points[i].x - int(x_cof[0])
        tp.y = points[i].y - int(y_range[0])
        pointss.append(tp)
    return pointss

def calc_cut_epointsR(points, x_cof, y_range):
    '''
    计算切割后的核线影像坐标
    :param points:
    :param x_cof:
    :param y_range:
    :return:
    '''
    pointss = []
    for i in range(len(points)):
        tp = point()
        tp.x = points[i].x - int(x_cof[1])
        tp.y = points[i].y - int(y_range[0])
        pointss.append(tp)
    return pointss

def Cal_epipolar_geotrans_L(dataset, Grids, x_cof, y_range):
    '''
    第一步：计算原始影像上四个角点地理坐标
    第二步：计算原始影像上四个角点在核线影像上影像坐标
    第三步：根据地理坐标和影像坐标利用最小二乘计算仿射变化矩阵的参数
    :return:
    '''
    trans_original = dataset.GetGeoTransform()
    img = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
    points_geo = Cal_cornerpoint_geo_cor(trans_original, img)
    points = []
    points.append(point(0,0))
    points.append(point(0,img.shape[1]))
    points.append(point(img.shape[0], 0))
    points.append(point(img.shape[0], img.shape[1]))
    points_epipolar = Cal_trans_fromCGPS(Grids, points)
    points_epipolar = calc_cut_epointsL(points_epipolar,x_cof, y_range)
    Trans = Aff_xy(points_epipolar, points_geo)
    return Trans

def Cal_epipolar_geotrans_R(dataset, Grids, x_cof, y_range):
    '''
    第一步：计算原始影像上四个角点地理坐标
    第二步：计算原始影像上四个角点在核线影像上影像坐标
    第三步：根据地理坐标和影像坐标利用最小二乘计算仿射变化矩阵的参数
    :return:
    '''
    trans_original = dataset.GetGeoTransform()
    img = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
    points_geo = Cal_cornerpoint_geo_cor(trans_original, img)
    points = []
    points.append(point(0,0))
    points.append(point(0,img.shape[1]))
    points.append(point(img.shape[0], 0))
    points.append(point(img.shape[0], img.shape[1]))
    points_epipolar = Cal_trans_fromCGPS(Grids, points)
    points_epipolar = calc_cut_epointsR(points_epipolar,x_cof, y_range)
    Trans = Aff_xy(points_epipolar, points_geo)
    return Trans

if __name__ == "__main__":
    p1 = point(0, 0)
    p2 = point(99, 190)
    p3 = point(2, 45)
    p4 = point(44, 99)
    ps = [p1, p2, p3, p4]
    m = Deterimin_in_polygon(p1,p2, p3, p4, point(49.5, 95))
    print(m)
