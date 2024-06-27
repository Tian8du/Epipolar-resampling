import math

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 自己定义的像素点类
class point:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)


def func(x, p):
    k,b = p
    return k*x+b

def residuals(p, y, x):
    return y - func(x, p)

def func1(x,y, p):
    a1, a2, a3, a4, a5, a6  = p
    re = a1 + a2* x + a3*y + a4*x*x + a5*x*y + a6*y*y
    return re

def residuals2(p, m, x, y ):
    temp = func1(x,y, p)

    return result

def eastsq_fit_curve(old_images, new_images):
    xy2_x = []
    xy2_y = []
    xy_x = []
    xy_y = []
    for i in range(len(old_images)):
        xy_x.append(new_images[i].x)
        xy_y.append(new_images[i].y)
        xy2_x.append(old_images[i].x)
        xy2_y.append(old_images[i].y)
    p0 = np.matrix([[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])
    xy_x = np.asarray(xy_x)
    xy_y = np.asarray(xy_y)
    X = np.vstack((xy_x, xy_y))
    xy2_x = np.asarray(xy2_x)
    xy2_y = np.asarray(xy2_y)
    Y = np.vstack((xy2_x, xy2_y))
    plsq = leastsq(residuals2,p0, args=(X, Y))
    return plsq


# 最小二乘拟合直线
def eastsq_fit(points):
    X = []
    Y = []
    # Step1: value the X and Y
    for i in range(len(points)):
        X.append(points[i].x)
        Y.append(points[i].y)
    p0=[0, 0]
    X = np.asarray(X)
    Y = np.asarray(Y)
    plsq = leastsq(residuals, p0, args=(Y, X))
    return plsq

# 获取与已知直线垂直的直线
def get_perpendicular(k, b, point):
    '''
    Get the perpendicular line of one straight line.
    :param k:
    :param b:
    :param point:
    :return:
    '''
    gradient = -1 /k
    intercept = point.y + point.x / k
    return gradient, intercept

# 获取直线和矩阵相交的范围,获取核线的范围
def get_area(k, b, data, inter_dis ):
    x_max = data.shape[1]
    y_max = data.shape[0]
    cen_point = point()
    cen_point.x = x_max/2
    cen_point.y = y_max/2
    dis = math.sqrt(x_max * x_max + y_max * y_max)
    m = dis / 2
    m = int(m / inter_dis) + 1
    angle = math.atan(k)
    # angle = math.pi - angle
    points = []
    for i in range(1, m):

        temp1 = point()
        temp2 = point()
        temp1.x = cen_point.x - inter_dis * math.cos(angle) * i
        temp1.y = temp1.x * k + b
        temp2.x = cen_point.x + inter_dis * math.cos(angle) * i
        temp2.y = temp2.x * k + b

        points.append(temp1)
        points.append(temp2)
    points.append(cen_point)


    mm = points[:]
    mm.sort(key=lambda elem: elem.x)
    return mm


def Indirect_Adjustment(B, P, L):
    """
    This is a funcction to realize the Indirecti Adjustment
    :param B: Matrix B
    :param P: Matrix P
    :param L: Matrix L
    :return: Matrix x
    """
    Nbb = np.dot(np.dot(B.T,P),B)
    Nbb_ = np.linalg.inv(Nbb)
    W = np.dot(np.dot(B.T,P),L)
    x= np.dot(Nbb_, W)
    return x


def eastsq_fit_affine(original_pts, temp_pts):
    '''

    :param original_pts:
    :param temp_pts:
    :return:
    '''
    x = original_pts[0].x
    y = original_pts[0].y
    x_ = temp_pts[0].x - 0
    y_ = temp_pts[0].y - 0
    B = np.matrix([1, x, y, x*x, x*y, y*y, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, x, y, x*x, x*y, y*y])
    L = np.matrix([x_],[y_])
    P = np.eye(len(original_pts))
    X = np.zeros(len(original_pts),1)
    for i in range(len(original_pts)):
        if(B.shape[0] == 2):
            pass
        else:
            temp_x = original_pts[i].x
            temp_y = original_pts[i].y
            temp_x_ = temp_pts[i].x - 0
            temp_y_ = temp_pts[i].y - 0
            temp_B = np.matrix([1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y])
            temp_L = np.matrix([temp_x_], [temp_y_])
            B = np.vstack((B, temp_B))
            L = np.vstack((L, temp_L))
    vx=Indirect_Adjustment(B,P,L)
    X = X + vx
    while(np.linalg.norm(vx,ord=2,axis=0) <1e-10):
        x = original_pts[0].x
        y = original_pts[0].y
        x_ = temp_pts[0].x
        y_ = temp_pts[0].y
        B = np.matrix([1, x, y, x * x, x * y, y * y, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, x, y, x * x, x * y, y * y])
        L = np.matrix([x_], [y_])
        P = np.eye(len(original_pts))
        X = np.zeros(12, 1)
        for i in range(len(original_pts)):
            if (B.shape[0] == 2):
                pass
            else:
                temp_x = original_pts[i].x
                temp_y = original_pts[i].y
                temp_x_ = temp_pts[i].x - 0
                temp_y_ = temp_pts[i].y - 0
                temp_B = np.matrix(
                    [1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y])
                temp_L = np.matrix([temp_x_], [temp_y_])
                B = np.vstack((B, temp_B))
                L = np.vstack((L, temp_L))
        vx = Indirect_Adjustment(B, P, L)
        X = X + vx

def leasq_matrix(X, Y):
    '''
    This is a function to calculate the leatsq matrix: BX = Y
    :param X:
    :param Y:
    :return:
    '''
    result = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return result


def cal_coff_aff(original_pts, temp_pts):
    '''
    Calulate the aff coff of two images by some points
    :param original_pts:
    :param temp_pts:
    :return:
    '''
    x = original_pts[0].x
    y = original_pts[0].y
    x_ = temp_pts[0].x
    y_ = temp_pts[0].y
    X = np.matrix([[1, x, y, x * x, x * y, y * y, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, x, y, x * x, x * y, y * y]])
    Y = np.matrix([[x_],[y_]])
    for i in range(len(original_pts)):
        if(i == 0):
            pass
        else:
            temp_x = original_pts[i].x
            temp_y = original_pts[i].y
            temp_x_ = temp_pts[i].x - 0
            temp_y_ = temp_pts[i].y - 0
            temp_X = np.matrix([[1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y]])
            temp_Y = np.matrix([[temp_x_], [temp_y_]])
            X = np.vstack((X, temp_X))
            Y = np.vstack((Y, temp_Y))

    B = leasq_matrix(X, Y)
    M = np.dot(X,B)
    return B
def sort_points(points):
    '''
    Sort points by the x of point
    :param points: list of points
    :return: ordered list of points
    '''
    mm = points[:]
    mm.sort(key=lambda elem:elem.x)
    return mm

def sort_points_y(points):
    '''
    Sort points by the x of point
    :param points: list of points
    :return: ordered list of points
    '''
    mm = points[:]
    mm.sort(key=lambda elem:elem.y)
    return mm

def cal_coff_aff3(original_pts, temp_pts):
    '''
    Calulate the aff coff of two images by some points
    :param original_pts:
    :param temp_pts:
    :return:
    '''
    x = original_pts[0].x
    y = original_pts[0].y
    x_ = temp_pts[0].x
    y_ = temp_pts[0].y
    X = np.matrix([[1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y]])
    Y = np.matrix([[x_],[y_]])
    for i in range(len(original_pts)):
        if(i == 0):
            pass
        else:
            temp_x = original_pts[i].x
            temp_y = original_pts[i].y
            temp_x_ = temp_pts[i].x - 0
            temp_y_ = temp_pts[i].y - 0
            temp_X = np.matrix([[1, temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y,
                                 temp_x * temp_x * temp_x, temp_x * temp_x * temp_y, temp_x *temp_y * temp_y,
                                 temp_y * temp_y * temp_y,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           temp_x, temp_y, temp_x * temp_x, temp_x * temp_y, temp_y * temp_y,
                           temp_x * temp_x * temp_x, temp_x * temp_x * temp_y, temp_x * temp_y * temp_y,
                           temp_y * temp_y * temp_y]])
            temp_Y = np.matrix([[temp_x_], [temp_y_]])
            X = np.vstack((X, temp_X))
            Y = np.vstack((Y, temp_Y))

    B = leasq_matrix(X, Y)
    M = np.dot(X,B)
    return B

def get_changed_xy(x, y, cof):
    '''

    :param point:
    :param cof:
    :return:
    '''
    temppt = point()
    a1, a2, a3, a4, a5, a6 = cof[0,0], cof[1,0], cof[2,0], cof[3,0], cof[4,0], cof[5,0]
    b1, b2, b3, b4, b5, b6 = cof[6,0], cof[7,0], cof[8,0], cof[9,0], cof[10,0], cof[11,0]
    temppt.x = a1 + a2 * x + a3 * y + a4 * x * x + a5 * x * y + a6 * y * y
    temppt.y = b1 + b2 * x + b3 * y + b4 * x * x + b5 * x * y + b6 * y * y
    return temppt

def get_changed_xy3(x, y, cof):
    '''

    :param point:
    :param cof:
    :return:
    '''
    temppt = point()
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = cof[0, 0], cof[1, 0], cof[2, 0], cof[3, 0], cof[4, 0], cof[5, 0], cof[6, 0], cof[7, 0], cof[8, 0], cof[9, 0]
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 = cof[10, 0], cof[11, 0], cof[12, 0], cof[13, 0], cof[14, 0], cof[15, 0], cof[16, 0], cof[17, 0], cof[18, 0], cof[19, 0]

    temppt.x = a1 + a2 * x + a3 * y + a4 * x * x + a5 * x * y + a6 * y * y + \
               a7 * x * x * x + a8 * x * x * y + a9 * x * y * y + a10 * y * y * y
    temppt.y = b1 + b2 * x + b3 * y + b4 * x * x + b5 * x * y + b6 * y * y + \
               b7 * x * x * x + b8 * x * x * y + b9 * x * y * y + b10 * y * y * y

    return temppt


def get_changed_xy_fast(x, y, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6):
    '''
    fast get xy
    :param x:
    :param y:
    :param a1:
    :param a2:
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param b5:
    :param b6:
    :return:
    '''
    temppt = point()
    temppt.x = a1 + a2 * x + a3 * y + a4 * x * x + a5 * x * y + a6 * y * y
    temppt.y = b1 + b2 * x + b3 * y + b4 * x * x + b5 * x * y + b6 * y * y
    return temppt


def cal_position2(point, x_max, y_max):
    '''
    Judge whether the point is in the plane
    :param point:
    :param img:
    :return:
    '''
    if (point.x >= x_max-1 or point.x <= 1 or point.y >= y_max-2 or point.y <= 2 ):
        return 0
    else:
        return 1

def cal_position(point, img):
    x_max = img.shape[1]
    y_max = img.shape[0]
    if (point.x >= x_max-1 or point.x <= 1 or point.y >= y_max-2 or point.y <= 2 ):
        return 0
    else:
        return 1


def extra_xy(points):
    '''
    Extra x and y from some points
    :param points: just the point
    :return: list of Xs and list of Ys
    '''
    X = []
    Y = []
    for i in range(len(points)):
        X.append(points[i].x)
        Y.append(points[i].y)
    return X, Y

def extra_points_from_list(list_points):
    points = []
    for i in range(len(list_points)):
        for j in range(len(list_points[i])):
            points.append(list_points[i][j])
    return points

def extra_xys(pointss):
    '''
    Extra x and y form list of points
    :param pointss:
    :return:
    '''
    X = []
    Y = []
    for i in range(len(pointss)):
        for j in range(len(pointss[i])):
            X.append(pointss[i][j].x)
            Y.append(pointss[i][j].y)
    return X, Y

def bilinear_interpolation(point, img):
    '''
    双线性内插经典算法，给定一个点point的位置，在img中内插出这个点的灰度值
    :param point: input  point with xy
    :param img:  Matrix orray indicates a image
    :return:  calucated value of the input points
    '''
    # x是列坐标，y是行坐标
    src_y = point.x
    src_x = point.y
    y_max = img.shape[1]
    x_max = img.shape[0]
    if (src_x > (x_max-2) or src_x < 2 or src_y >(y_max-2) or src_y < 2 ):
        return 0
    src_x0 = int(src_x)
    src_y0 = int(src_y)
    src_x1 = src_x0 + 1
    src_y1 = src_y0 + 1
    # 双线性插值
    temp0 = (src_x1 - src_x) *(src_y1-src_y) * img[src_x0][src_y0] + (src_x - src_x0) * (src_y1-src_y) * img[src_x1][src_y0]
    temp1 = (src_x1 - src_x)* (src_y-src_y0) * img[src_x0][src_y1] + (src_x - src_x0) * (src_y-src_y0) * img[src_x1][src_y1]
    value =  temp0 +  temp1
    return value

def cal_region(img, coef):
    '''
    Get the max x and y after interpolation
    A---------B
    -----------
    -----------
    C---------D
    :param img: image --Matrix
    :param rpc: rational pp cof
    :return: max x and y
    '''
    pointA = point()
    pointB = point()
    pointC = point()
    pointD = point()
    pointA = get_changed_xy(0, 0, coef)
    pointB = get_changed_xy(0, img.shape[1], coef)
    pointC = get_changed_xy(img.shape[0], 0, coef)
    pointD = get_changed_xy(img.shape[0], img.shape[1], coef)
    max_x = max(pointA.x, pointB.x, pointC.x, pointD.x)
    min_x = min(pointA.x, pointB.x, pointC.x, pointD.x)
    max_y = max(pointA.y, pointB.y, pointC.y, pointD.y)
    min_y = min(pointA.y, pointB.y, pointC.y, pointD.y)
    return min_x, max_x, min_y, max_y

def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)

def Deterimin_in_polygon(point1, point2, point3, point4, pointp):
    '''
    判断point是否在由四个点按照顺时针顺序组成的凸多边形内
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :param point:
    :return:
    '''
    a = GetCross(point1, point2, pointp)
    b = GetCross(point2, point3, pointp)
    c = GetCross(point3, point4, pointp)
    d = GetCross(point4, point1, pointp)
    if ( a > 0 and b > 0 and c > 0 and d > 0) or ( a<0 and b <0 and c < 0 and d < 0):
        return 1
    elif ( a == 0 or b == 0 or c == 0 or d == 0):
        return 1
    else:
        return 0
def Deterimin_in_polygon2(point1, point2, point3, point4, pointp):
    '''
    判断point是否在由四个点按照顺时针顺序组成的凸多边形内
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :param point:
    :return:
    '''
    a = cal_triangle_area(point1, point2, pointp)
    b = cal_triangle_area(point2, point3, pointp)
    c = cal_triangle_area(point3, point4, pointp)
    d = cal_triangle_area(point4, point1, pointp)
    s4 = a + b + c + d
    s = cal_rec_Area(point1, point2, point3, point4)
    if ( s4 == s):
        return 1
    else:
        return 0

def cal_triangle_area(point1, point2, point3):
    '''
    计算给定三个点的组成三角形的面积
    :param point1:
    :param point2:
    :param point3:
    :return:
    '''
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    x3, y3 = point3.x, point3.y
    m = (x1-x3)*(y2-y3)- (x2-x3)*(y1-y3)
    m = abs(m)
    m = m/2.0
    return m

def cal_rec_Area(point1, point2, point3, point4):
    '''
    计算四边形面积
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :return:
    '''
    d12 = calcDistance(point1, point2)
    d23 = calcDistance(point2, point3)
    d34 = calcDistance(point3, point4)
    d41 = calcDistance(point4, point1)
    d24 = calcDistance(point2, point4)
    k1 = (d12+d41+d24)/2
    k2 = (d23+d34+d24)/2
    s1 = (k1*(k1-d12)*(k1-d41)*(k1-d24))**0.5
    s2 = (k2*(k2-d23)*(k2-d34)*(k2-d24))**0.5
    s = s1+s2
    return s

def calcDistance(point1, point2):
    '''
    计算两点间距离
    :param point1:
    :param point2:
    :return:
    '''
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    dis = math.sqrt(((x1-x2)**2 + (y1-y2)**2))
    return dis

def Aff_xy(original_pts, temp_pts):
    '''
    Calulate the aff coff of two images by four points
    :param original_pts:
    :param temp_pts:
    :return:
    '''
    x = original_pts[0].x
    y = original_pts[0].y
    x_ = temp_pts[0].x
    y_ = temp_pts[0].y
    X = np.matrix([[1, x, y, 0, 0, 0],[0, 0, 0, 1, x, y]])
    Y = np.matrix([[x_],[y_]])
    for i in range(len(original_pts)):
        if(i == 0):
            pass
        else:
            temp_x = original_pts[i].x
            temp_y = original_pts[i].y
            temp_x_ = temp_pts[i].x - 0
            temp_y_ = temp_pts[i].y - 0
            temp_X = np.matrix([[1, temp_x, temp_y,  0, 0, 0],
                          [ 0, 0, 0, 1, temp_x, temp_y]])
            temp_Y = np.matrix([[temp_x_], [temp_y_]])
            X = np.vstack((X, temp_X))
            Y = np.vstack((Y, temp_Y))

    B = leasq_matrix(X, Y)
    M = np.dot(X,B)
    return B

def get_distance_point2line(point, line_ab):
    """
    Args:
        point: [x0, y0]
        line_ab: [k, b]
    """
    k, b = line_ab
    distance = abs(k * point[0] - point[1] + b) / math.sqrt(k**2 + 1)
    return distance

def cal_RMS(list):
    sum = 0
    for i in range(len(list)):
        sum = sum + list[i]
    ave = sum/len(list)

    RMS2  = 0
    for i in range(len(list)):
        temp = (list[i] - ave)**2
        RMS2 = RMS2 + temp
    RMS2 = RMS2/len(list)
    RMS = math.sqrt(RMS2)
    return RMS

def test5poitns(x_max, y_max):
    points = []
    point1 = [int(x_max/4), int(y_max/4)]
    point2 = [int(x_max/4), int(y_max/4*3)]
    point3 = [int(x_max/4*3), int(y_max/4*3)]
    point4 = [int(x_max/4*3),int(y_max/4)]
    point5 = [int(x_max/2), int(y_max/2)]
    points = [point1, point2, point3, point4, point5]
    return points

if __name__ == "__main__":
    point1 = point()
    point1.x =0
    point1.y =10
    point2 = point()
    point2.x=10
    point2.y=10
    point3 = point()
    point3.x = 10
    point3.y = 0
    point4 = point()
    point4.x = 0
    point4.y = 0
    pointp = point()
    pointp.x = 17.9
    pointp.y = 4.5

    s = Deterimin_in_polygon2(point1,point2,point3,point4,point(9.99999999,5))
    print(s)









