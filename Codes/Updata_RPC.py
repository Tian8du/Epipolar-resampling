import numpy as np
from rpc import RPC_Positive
from Basefunction import Deterimin_in_polygon

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from Basefunction import Aff_xy, point

# Step1: 获取物方空间上分布均匀的控制点
# Step2：利用RPC公式投影到原始影像上
# Step3：利用核线影像和原始影像间转换关系得到核线控制点坐标
# Step4：利用最小二乘原理计算核线影像的RPC系数

class geo_point:
    '''
    x0,y0是原始影像坐标，x1,y1是核线影像坐标
    '''
    def __init__(self,lon=0, lat=0, h=0, x0=0, y0=0, x1=0, y1=0):
        self.lon = float(lon)
        self.lat = float(lat)
        self.h = h
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    # 正则化点位信息,
    def Normalize(self, rpc):
        self.U = (self.lat - rpc.latOffset) / rpc.latScale
        self.V = (self.lon - rpc.LongOffset) / rpc.LongScale
        self.W = (self.h - rpc.heightOffset) / rpc.heightScale
        self.Y = (self.y1 - rpc.lineOffset) / rpc.lineScale
        self.X = (self.x1 - rpc.sampOffset) / rpc.sampScale

def Initial_GCPS(lon_range, lat_range, h_range):
    '''
    获取初始化物方控制点，控制点在空间上均匀分布
    :param lon_range: longtitude range, lon_range[0]:Start, lon_range[1]:end
    :param lat_range: latigude range, lat_range[0]:Start, lat_rangep1]:end
    :param h_range: height range, h_range[0]:Start, h_range[1]:end
    :return:
    '''
    points = np.zeros((5, 5, 5), dtype=geo_point)
    for i in np.arange(lon_range[0], lon_range[1], (lon_range[0], lon_range[1])/5):
        for j in np.arange(lat_range[0], lat_range[1], (lat_range[1]- lat_range[1])/5):
            for h in np.range(h_range[0], h_range[1], (h_range[1]-h_range[0])/5):
                points[i, j, h] = geo_point(i, j, h)
    return points

def Initial_original_points(points):
    '''
    初始化原始影像上的坐标点
    :param points: class geo_points
    :return:
    '''
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            for h in range(points.shape[2]):
                points[i, j, h].y, points[i, j, h].x = RPC_Positive(points[i, j, h].lon, points[i, j, h].lat, points[i, j, h].h)
    return points

def Initial_new_points(points, Grids):
    '''
    计算核线影像上的控制点影像坐标
    :param points:
    :param Grids:
    :return:
    '''
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            for h in range(points.shape[2]):
                for g in range(len(Grids)):
                    pointp = points[i, j, h]
                    pointA = Grids[g].grid_old.point1
                    pointB = Grids[g].grid_old.point2
                    pointC = Grids[g].grid_old.point4
                    pointD = Grids[g].grid_old.point3
                    if (Deterimin_in_polygon(pointA, pointB, pointC, pointD, pointp) == 1):
                        a1, a2, a3, b1, b2, b3 = Grids[g].cof[0:6]
                        a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)
                        points[i, j, h].y = (a2 * pointp.y - a2 * b1 - b2 * pointsp.x + a1 * b2) / (a2 * b3 - a3 * b2)
                        points[i, j, h].x = (pointp.x - a1 - a3 * tp.y) / a2
    return points

def Calc_coffs(points, rpc):
    '''
    根据地面点和核线点利用RPC公式和最小二乘原理计算RPC的系数
    :param points:
    :return:
    '''
    # Step1: 归一化处理
    for i in len(points):
        poitns[i].Normalize(rpc)

    # Step2: V = Bx - L
    U, V , W, Y, X = points[0].U, points[0].V, points[0].W, points[0].Y, points[0].X
    B = np.matrix([[1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                    U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                    U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                    - X, - X * V, - X * U, - X * W, - X * V * U, - X * V * W, - X * U * W, - X * V * V, - X * U * U,
                    - X * W * W,
                    - X * U * V * W, - X * V * V * V, - X * V * U * U, - X * V * W * W, - X * V * V * U,
                    - X * U * U * U, - X * U * W * W, - X * V * V * W, - X * U * U * W, - X * W * W * W,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                    U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                    U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                    - Y, - Y * V, - Y * U, - Y * W, - Y * V * U, - Y * V * W, -Y * U * W, - Y * V * V, - Y * U * U,
                    - Y * W * W,
                    - Y * U * V * W, - Y * V * V * V, - Y * V * U * U, - Y * V * W * W, - Y * V * V * U,
                    - Y * U * U * U, - Y * U * W * W, - Y * V * V * W, - Y * U * U * W, - Y * W * W * W,
                   ]])
    x = np.matrix([rpc.lineNumCoef, rpc.lineDenCoef, rpc.sampNumCoef, rpc.sampDenCoef])
    l = np.dot(B, x)
    for i in len(points):
        if (i == 0):
            pass
        else:
            U, V, W, Y, X = points[i].U, points[i].V, points[i].W, points[i].Y, points[i].X
            temp_B = np.matrix([[1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                            U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                            U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                            - X, - X * V, - X * U, - X * W, - X * V * U, - X * V * W, - X * U * W, - X * V * V,
                            - X * U * U,
                            - X * W * W,
                            - X * U * V * W, - X * V * V * V, - X * V * U * U, - X * V * W * W, - X * V * V * U,
                            - X * U * U * U, - X * U * W * W, - X * V * V * W, - X * U * U * W, - X * W * W * W,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                            U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                            U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                            - Y, - Y * V, - Y * U, - Y * W, - Y * V * U, - Y * V * W, -Y * U * W, - Y * V * V,
                            - Y * U * U,
                            - Y * W * W,
                            - Y * U * V * W, - Y * V * V * V, - Y * V * U * U, - Y * V * W * W, - Y * V * V * U,
                            - Y * U * U * U, - Y * U * W * W, - Y * V * V * W, - Y * U * U * W, - Y * W * W * W,
                            ]])
            # x = np.matrix([rpc.lineNumCoef, rpc.lineDenCoef, rpc.sampNumCoef, rpc.sampDenCoef])
            temp_l = np.dot(temp_B, x)
            B = np.vstack((B, temp_B))
            l = np.vstack((l, temp_l))
            x = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(l)
        Ns = x[0:20]
        Ds = x[20:40]
        Nl = x[40:60]
        Ns = x[60:80]
    return Ns, Ds, Nl, Ns

def leasq_20cof(points, rpc):
    U, V, W, Y, X = points[0].U, points[0].V, points[0].W, points[0].Y, points[0].X
    B = np.matrix([[1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                    U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                    U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                    - X, - X * V, - X * U, - X * W, - X * V * U, - X * V * W, - X * U * W, - X * V * V, - X * U * U,
                    - X * W * W,
                    - X * U * V * W, - X * V * V * V, - X * V * U * U, - X * V * W * W, - X * V * V * U,
                    - X * U * U * U, - X * U * W * W, - X * V * V * W, - X * U * U * W, - X * W * W * W,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                    U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                    U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                    - Y, - Y * V, - Y * U, - Y * W, - Y * V * U, - Y * V * W, -Y * U * W, - Y * V * V, - Y * U * U,
                    - Y * W * W,
                    - Y * U * V * W, - Y * V * V * V, - Y * V * U * U, - Y * V * W * W, - Y * V * V * U,
                    - Y * U * U * U, - Y * U * W * W, - Y * V * V * W, - Y * U * U * W, - Y * W * W * W,
                    ]])
    x = np.matrix([rpc.lineNumCoef, rpc.lineDenCoef, rpc.sampNumCoef, rpc.sampDenCoef])
    l = np.dot(B, x)
    for i in len(points):
        if (i == 0):
            pass
        else:
            U, V, W, Y, X = points[i].U, points[i].V, points[i].W, points[i].Y, points[i].X
            temp_B = np.matrix([[1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                                 U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                                 U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                                 - X, - X * V, - X * U, - X * W, - X * V * U, - X * V * W, - X * U * W, - X * V * V,
                                 - X * U * U,
                                 - X * W * W,
                                 - X * U * V * W, - X * V * V * V, - X * V * U * U, - X * V * W * W, - X * V * V * U,
                                 - X * U * U * U, - X * U * W * W, - X * V * V * W, - X * U * U * W, - X * W * W * W,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, V, U, W, V * U, V * W, U * W, V * V, U * U, W * W,
                                 U * V * W, V * V * V, V * U * U, V * W * W, V * V * U,
                                 U * U * U, U * W * W, V * V * W, U * U * W, W * W * W,
                                 - Y, - Y * V, - Y * U, - Y * W, - Y * V * U, - Y * V * W, -Y * U * W, - Y * V * V,
                                 - Y * U * U,
                                 - Y * W * W,
                                 - Y * U * V * W, - Y * V * V * V, - Y * V * U * U, - Y * V * W * W, - Y * V * V * U,
                                 - Y * U * U * U, - Y * U * W * W, - Y * V * V * W, - Y * U * U * W, - Y * W * W * W,
                                 ]])
            # x = np.matrix([rpc.lineNumCoef, rpc.lineDenCoef, rpc.sampNumCoef, rpc.sampDenCoef])
            temp_l = np.dot(temp_B, x)
            B = np.vstack((B, temp_B))
            l = np.vstack((l, temp_l))
            x = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(l)
    return x

def ReadXML(xmlfile):
    """
    get information from xml file.
    :param xmlfile: XML file of GF7 image
    :return: Return the GeoTrans of corrospending img
    1       2
        3
    4       5
    """

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    Width = float(root[28].text)
    Height = float(root[29].text)
    p3 = geo_point(root[60].text, root[59].text, 0, Width/2, Height/2)
    p1 = geo_point(root[62].text, root[61].text, 0, 0, 0)
    p2 = geo_point(root[64].text, root[63].text, 0, Width, 0)
    p4 = geo_point(root[68].text, root[67].text, 0, 0, Height)
    p5 = geo_point(root[66].text, root[65].text, 0, Width, Height)

    img_p3 = point(Width/2, Height/2)
    cor_p3 = point(root[60].text, root[59].text)
    img_p1 = point(0, 0)
    cor_p1 = point(root[62].text, root[61].text)
    img_p2 = point(Width, 0)
    cor_p2 = point(root[64].text, root[63].text)
    img_p4 = point(0, Height)
    cor_p4 = point(root[68].text, root[67].text)
    img_p5 = point(Width, Height)
    cor_p5 = point(root[66].text, root[65].text)

    img_points = [img_p1, img_p2, img_p3, img_p4, img_p5]
    cor_points = [cor_p1, cor_p2, cor_p3, cor_p4, cor_p5]
    GeoTrans = Aff_xy(img_points, cor_points)
    # for child in root:
    #     print( child.tag, child.text)
    #     # print("tag:", child.text)
    #     # print("attrib:", child.attrib)
    #     # child.set("set:","设置属性")
    return GeoTrans



if __name__ == "__main__":
    xmlfile = r"K:\GF7\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.xml"
    mm = ReadXML(xmlfile)



