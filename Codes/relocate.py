import numpy as np
import math
import Basefunction
from Basefunction import point
from copy import deepcopy
from Basefunction import sort_points


# 逆时针旋转一个点
def point_rotate(angle, cenpoint, other_point):
    '''
    Rotate a point anticlockwise
    :param angle: just the angle(rad)
    :param cenpoint: the point surrounded
    :param other_point: the point rotating
    :return: the point that rotated
    '''
    temp = point()
    temp.x = cenpoint.x + (other_point.x - cenpoint.x) * math.cos(angle) - (other_point.y - cenpoint.y) * math.sin(
        angle)
    temp.y = cenpoint.y + (other_point.y - cenpoint.y) * math.cos(angle) + (other_point.x - cenpoint.x) * math.sin(
        angle)
    return temp


# 以 cenpoint 为中心,将line逆时针旋转angle角度
def line_rotate(angle, cenpoint, line):
    templine = []
    for i in range(len(line)):
        temp_point = point_rotate(angle, cenpoint, line[i])
        templine.append(temp_point)
    return templine


# 以 cenpoint 为中心，将liens 逆时针旋转angle 角度, 并且把点进行平移到正数范围内
def lines_rotate(angle, cenpoint, lines):
    templines = []
    for i in range(len(lines)):
        templine = line_rotate(angle, cenpoint, lines[i])
        templines.append(templine)
    return templines


def change_y_line(L_points, R_points):
    L = []
    R = []
    temp1 = point()
    temp2 = point()
    y = L_points[0].y
    for i in range(len(L_points)):
        temp1.x = L_points[i].x
        temp1.y = y
        mm = deepcopy(temp1)
        L.append(mm)
    for j in range(len(R_points)):
        temp2.x = R_points[j].x
        temp2.y = y
        nn = deepcopy(temp2)
        R.append(nn)
    return L, R


def change_y_lines(L_images, R_images):
    L = []
    R = []
    for i in range(len(L_images)):
        L_line, R_line = change_y_line(L_images[i], R_images[i])
        tempL = deepcopy(L_line)
        tempR = deepcopy(R_line)
        L.append(tempL)
        R.append(tempR)
    return L, R


def xy_normolize(points):
    Xs = []
    Ys = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            Xs.append(points[i][j].x)
            Ys.append(points[i][j].y)
    minx, miny = min(Xs), min(Ys)
    for i in range(len(points)):
        for j in range(len(points[i])):
            points[i][j].x = points[i][j].x - minx
            points[i][j].y = points[i][j].y - miny
    Xs = []
    Ys = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            Xs.append(points[i][j].x)
            Ys.append(points[i][j].y)
    minxx, minyy = min(Xs), min(Ys)
    return points


def change_x_line(points, p_dis, x1):
    '''
    change x distance making dx the same
    :param points: list of points from orignal image
    :return: standard grid points
    '''

    for i in range(len(points)):
        points[i].x = x1 + i * p_dis

    return points


def change_x_lines(pointss):
    '''

    :param pointss:
    :return:
    '''
    for i in range(len(pointss)):
        pointss[i] = sort_points(pointss[i])

    num = int(len(pointss) / 2)
    x1 = pointss[num][0].x
    y1 = pointss[num][0].y
    x2 = pointss[num][-1].x
    y2 = pointss[num][-1].y
    dis = math.sqrt((x1 - x2) ** 2 + (y1 - y1) ** 2)
    point_num = len(pointss[num])
    p_dis = dis / (point_num - 1)

    for i in range(len(pointss)):
        pointss[i] = change_x_line(pointss[i], p_dis, x1)

    return pointss, p_dis


def change_x_lines_R(pointss, p_dis):
    '''

    :param pointss:
    :return:
    '''
    for i in range(len(pointss)):
        pointss[i] = sort_points(pointss[i])

    num = int(len(pointss) / 2)
    x1 = pointss[num][0].x

    for i in range(len(pointss)):
        pointss[i] = change_x_line(pointss[i], p_dis, x1)

    return pointss


def move_y(R_points, dis):
    for i in range(len(R_points)):
        for j in range(len(R_points[i])):
            R_points[i][j].y = R_points[i][j].y + dis
    return R_points


def move_x(R_points, dis):
    for i in range(len(R_points)):
        for j in range(len(R_points[i])):
            R_points[i][j].x = R_points[i][j].x + dis
    return R_points


def cen_align(L_cpt, R_cpt, L_points, R_points):
    '''
    Align the center point of Left img and Right img(same row)
    :param L_cpt:
    :param R_cpt:
    :param L_points:
    :param R_points:
    :return:
    '''
    disx = R_cpt.x - L_cpt.x
    disy = R_cpt.y - L_cpt.y
    for i in range(len(R_points)):
        for j in range(len(R_points[i])):
            R_points[i][j].y - disy
            R_points[i][j].x - disx
    return R_points


def CorNOR(L_points, R_points):
    """
    坐标对齐，保证初始核线在同一行
    :param L_points:
    :param R_points:
    :return:
    """
    num = int(len(L_points) / 2)
    y1 = L_points[num][0].y
    y2 = L_points[0][0].y
    dis1 = y1 - y2

    num = int(len(R_points) / 2)
    y1 = R_points[num][0].y
    y2 = R_points[0][0].y
    dis2 = y1 - y2

    dis = dis1 - dis2
    return dis1, dis2, dis


def relocate_sub(L_cen_point, L_points, L_k, R_cen_point, R_points, R_k):
    '''
    The whole sub of relocate
    :param L_cen_point:
    :param L_points:
    :param L_k:
    :param R_cen_point:
    :param R_points:
    :param R_k:
    :return:
    '''
    angleL = -math.atan(L_k)
    m = lines_rotate(angleL, L_cen_point, L_points)
    angleR = -math.atan(R_k)
    n = lines_rotate(angleR, R_cen_point, R_points)

    mm, nn = change_y_lines(m, n)

    mmm, dis = change_x_lines(mm)
    nnn, dis = change_x_lines(nn)

    mmmm = xy_normolize(mmm)
    nnnn = xy_normolize(nnn)
    return mmmm, nnnn


def relocate_sub2(L_cen_point, L_points, L_k, R_cen_point, R_points, R_k):
    '''
    The whole sub of relocate
    :param L_cen_point:
    :param L_points:
    :param L_k:
    :param R_cen_point:
    :param R_points:
    :param R_k:
    :return:
    '''
    angleL = -math.atan(L_k)
    m = lines_rotate(angleL, L_cen_point, L_points)
    angleR = -math.atan(R_k)
    n = lines_rotate(angleR, R_cen_point, R_points)

    mm, nn = change_y_lines(m, n)

    mmm, p_dis = change_x_lines(mm)
    nnn = change_x_lines_R(nn, p_dis)
    # nnn = remove_y(nnn, 200)

    mmmm = xy_normolize(mmm)
    nnnn = xy_normolize(nnn)
    return mmm, nnn


def pre_relocate_sub(L_cen_point, L_points, L_k, L_part, R_cen_point, R_points, R_k, R_part):
    angleL = -math.atan(L_k)
    m = lines_rotate(angleL, L_cen_point, L_points)
    angleR = -math.atan(R_k)
    n = lines_rotate(angleR, R_cen_point, R_points)

    mm, nn = change_y_lines(m, n)
    return mm, nn
