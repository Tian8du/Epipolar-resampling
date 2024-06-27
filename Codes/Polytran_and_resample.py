from Basefunction import point
from Basefunction import cal_position
from Basefunction import sort_points
from Basefunction import sort_points_y
from copy import deepcopy
import numpy as np
from scipy.optimize import leastsq
from Basefunction import leasq_matrix
from relocate import point_rotate
from relocate import lines_rotate
import math
from Basefunction import bilinear_interpolation
import tqdm
from Basefunction import Aff_xy
import rpc


class grid:
    def __init__(self, point1=point(), point2=point(), point3=point(), point4=point()):
        """
        point1-------point2
        -               -
        point3-------point4
        """
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4


class grid2:
    def __init__(self, grid_old=grid(), grid_new=grid(), cof=[]):
        self.grid_old = grid_old
        self.grid_new = grid_new
        self.cof = cof


def Cal_Minpolygon(points):
    """
    Calculate the  smallest area Polygon consisting points
    :param points: area points
    :return: max and min cordinate of the ploygon,Xmin, Xmax, Ymin, Ymax
    """
    Xs = []
    Ys = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            Xs.append(points[i][j].x)
            Ys.append(points[i][j].y)
    return min(Xs), max(Xs), min(Ys), max(Ys)


def Cal_Minpolygon_fromlists(points):
    """
    Calculate the  smallest area Polygon consisting points
    :param points: area points
    :return: max and min cordinate of the ploygon,Xmin, Xmax, Ymin, Ymax
    """
    Xs = []
    Ys = []
    for i in range(len(points)):
        Xs.append(points[i].x)
        Ys.append(points[i].y)
    return min(Xs), max(Xs), min(Ys), max(Ys)


def Sift_points(pts_sifted, pts, cen_point, k):
    angle = -math.atan(k)
    # m -- more points
    n = lines_rotate(angle, cen_point, pts)

    minx, maxx, miny, maxy = Cal_Minpolygon(n)
    points = []
    for i in range(len(pts_sifted)):
        templine = []
        for j in range(len(pts_sifted[i])):
            pt2 = pts_sifted[i][j]
            pt = point_rotate(angle, cen_point, pt2)
            if (minx - 0.1) < pt.x < (maxx + 0.11) and (miny - 0.1) < pt.y < (maxy + 0.1):
                templine.append(pt2)
        if len(templine) != 0:
            points.append(templine)
    return points


# def sort_points(points):
#     '''
#     Sort points by the x of point
#     :param points: list of points
#     :return: ordered list of points
#     '''
#     mm = points[:]
#     mm.sort(key=lambda elem:elem.x)
#     return mm


def Normalize_grids2(old_points, new_points):
    '''
    Generage Grid including four poits
    :param old_points: original points in image
    :param new_points: relocated points in image
    :return: Grids
    '''
    grids = []
    old_points_sorted = []
    new_points_sorted = []
    # Sort the point of one epipolar line by the value of x
    for i in range(len(new_points)):
        old_points_sorted.append(sort_points_y(old_points[i]))
        new_points_sorted.append(sort_points(new_points[i]))

    for i in range(len(new_points) - 1):
        for j in range(len(new_points[i]) - 1):
            tempgrid = grid()
            tempgrid.point1 = new_points_sorted[i][j]
            tempgrid.point2 = new_points_sorted[i][j + 1]
            tempgrid.point3 = new_points_sorted[i + 1][j]
            tempgrid.point4 = new_points_sorted[i + 1][j + 1]

            tempgridd = grid()
            tempgridd.point1 = old_points_sorted[i][j]
            tempgridd.point2 = old_points_sorted[i][j + 1]
            tempgridd.point3 = old_points_sorted[i + 1][j]
            tempgridd.point4 = old_points_sorted[i + 1][j + 1]

            new = []
            old = []
            new.append(tempgrid.point1)
            new.append(tempgrid.point2)
            new.append(tempgrid.point3)
            new.append(tempgrid.point4)
            old.append(tempgridd.point1)
            old.append(tempgridd.point2)
            old.append(tempgridd.point3)
            old.append(tempgridd.point4)
            cof = Aff_xy(new, old)

            tempgrid2 = grid2(tempgridd, tempgrid, cof)

            grids.append(tempgrid2)

    return grids


def Normalize_grids(old_points, new_points):
    """
    Generage Grid including four poits
    :param old_points: original points in image
    :param new_points: relocated points in image
    :return: Grids
    """
    grids = []
    old_points_sorted = []
    new_points_sorted = []
    # Sort the point of one epipolar line by the value of x
    for i in range(len(new_points)):
        old_points_sorted.append(sort_points(old_points[i]))
        new_points_sorted.append(sort_points(new_points[i]))

    for i in range(len(new_points) - 1):
        for j in range(len(new_points[i]) - 1):
            tempgrid = grid()
            tempgrid.point1 = new_points_sorted[i][j]
            tempgrid.point2 = new_points_sorted[i][j + 1]
            tempgrid.point3 = new_points_sorted[i + 1][j]
            tempgrid.point4 = new_points_sorted[i + 1][j + 1]

            tempgridd = grid()
            tempgridd.point1 = old_points_sorted[i][j]
            tempgridd.point2 = old_points_sorted[i][j + 1]
            tempgridd.point3 = old_points_sorted[i + 1][j]
            tempgridd.point4 = old_points_sorted[i + 1][j + 1]

            new = []
            old = []
            new.append(tempgrid.point1)
            new.append(tempgrid.point2)
            new.append(tempgrid.point3)
            new.append(tempgrid.point4)
            old.append(tempgridd.point1)
            old.append(tempgridd.point2)
            old.append(tempgridd.point3)
            old.append(tempgridd.point4)
            cof = Aff_xy(new, old)

            tempgrid2 = grid2(tempgridd, tempgrid, cof)

            grids.append(tempgrid2)

    return grids


def Generate_grids(old_points, new_points):
    """
    Generate Grid including four poits
    :param old_points: original points in image
    :param new_points: relocated points in image
    :return: Grids
    """
    grids = []
    old_points_sorted = []
    new_points_sorted = []
    # Sort the point of one epipolar line by the value of x
    for i in range(len(new_points)):
        old_points_sorted.append(sort_points(old_points[i]))
        new_points_sorted.append(sort_points(new_points[i]))

    for i in range(len(new_points) - 1):
        for j in range(len(new_points[i]) - 1):
            tempgrid = grid()
            tempgrid.point1 = new_points_sorted[i][j]
            tempgrid.point2 = new_points_sorted[i][j + 1]
            tempgrid.point3 = new_points_sorted[i + 1][j]
            tempgrid.point4 = new_points_sorted[i + 1][j + 1]

            tempgridd = grid()
            tempgridd.point1 = old_points_sorted[i][j]
            tempgridd.point2 = old_points_sorted[i][j + 1]
            tempgridd.point3 = old_points_sorted[i + 1][j]
            tempgridd.point4 = old_points_sorted[i + 1][j + 1]
            tempgrid2 = grid2(tempgridd, tempgrid, cof=None)
            grids.append(tempgrid2)

    return grids


def Normalize_resolution(L_grids, R_grids):
    return 0


def Generate_aff(grids):
    for i in range(len(grids)):
        new = []
        old = []
        new.append(grids[i].grid_new.point1)
        new.append(grids[i].grid_new.point2)
        new.append(grids[i].grid_new.point3)
        new.append(grids[i].grid_new.point4)
        old.append(grids[i].grid_old.point1)
        old.append(grids[i].grid_old.point2)
        old.append(grids[i].grid_old.point3)
        old.append(grids[i].grid_old.point4)
        grids[i].cof = Aff_xy(new, old)
    return grids


def Grid_bilinear_interpolation(startx0, starty0, y_dis, x_dis, y_dis0, x_dis0, i, j):
    y = starty0 + i / y_dis * y_dis0
    x = startx0 + j / x_dis * x_dis0
    temp = point()
    temp.x = x
    temp.y = y
    return temp


def Grid_aff_interpolation(a1, a2, a3, b1, b2, b3, i, j):
    temppt = point()
    temppt.x = a1 + a2 * j + a3 * i
    temppt.y = b1 + b2 * j + b3 * i
    return temppt


def Poly2_interpolation(a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, x, y):
    temppt = point()
    temppt.x = a1 + a2 * x + a3 * y + a4 * x * x + a5 * x * y + a6 * y * y
    temppt.y = b1 + b2 * x + b3 * y + b4 * x * x + b5 * x * y + b6 * y * y
    return temppt


def Poly3_interpolation(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, x, y):
    temppt = point()
    temppt.x = a1 + a2 * x + a3 * y + a4 * x * x + a5 * x * y + a6 * y * y + a7 * x * x * x + a8 * x * x * y + a9 * x * y * y + a10 * y * y * y
    temppt.y = b1 + b2 * x + b3 * y + b4 * x * x + b5 * x * y + b6 * y * y + b7 * x * x * x + b8 * x * x * y + b9 * x * y * y + b10 * y * y * y
    return temppt


def Matrix_splicing(chunk, row, col):
    # 计算第一行
    col_chunk = chunk[0]
    for j in range(1, col):
        col_chunk = np.hstack((col_chunk, chunk[j]))
    img = np.array(list(col_chunk))

    for i in range(1, row):
        col_chunk = chunk[i * col]
        for j in range(1, col):
            col_chunk = np.hstack((col_chunk, chunk[i * col + j]))
        img = np.vstack((img, col_chunk))
    return img


def value_chunk(chunk, data, grids):
    """
    Valut each chunk
    :param chunk: the chunk part consisting the information of small data
    :param data: the temp matrix storage information
    :param grids: just the grids
    :return: temp matrix valued
    """
    for i in range(len(chunk)):
        startx = int(grids[i].grid_new.point3.x)
        starty = int(grids[i].grid_new.point3.y)
        x_dis = chunk[i].shape[1]
        y_dis = chunk[i].shape[0]
        # if (y_dis < 2 or x_dis < 2):
        print("chunk ", i)
        data[starty:starty + y_dis, startx:startx + x_dis] = chunk[i]
    return data


def cut_minpolygon_L(img, x_cof, y_range):
    """
    Cut the Minpolygon from the image
    :param img: input image bigger than output image
    :param poly: the minpolygon
    :return: output matrix
    """

    img = img[round(y_range[0]):round(y_range[1]), round(x_cof[0]):round(x_cof[0] + x_cof[2])]
    return img


def cut_minpolygon_R(img, x_cof, y_range):
    img = img[round(y_range[0]):round(y_range[1]), round(x_cof[1]):round(x_cof[1] + x_cof[2])]
    return img


def Determin_region(mmm, nnn):
    poly_L = Cal_Minpolygon_fromlists(mmm)
    poly_R = Cal_Minpolygon_fromlists(nnn)

    # Step2:Determin the Y_scope
    y_range = []
    miny = min(poly_L[2], poly_R[2])
    maxy = max(poly_L[3], poly_R[3])
    y_dis = maxy - miny
    y_range.append(miny)
    y_range.append(maxy)
    y_range.append(y_dis)

    # Step3:Determin the X_scope
    x_cof = []
    side_dis1 = poly_L[1] - poly_L[0]
    side_dis2 = poly_R[1] - poly_R[0]
    x_dis = max(side_dis1, side_dis2)
    L_minx = poly_L[0]
    R_minx = poly_R[0]
    x_cof.append(L_minx)
    x_cof.append(R_minx)
    x_cof.append(x_dis)
    print("The region of image is caculated OK !")
    return poly_L, poly_R, x_cof, y_range


def sift_grids(grids, data):
    '''
    get the grid tha in the original image
    :param grids:
    :return:
    '''
    gg = []
    for i in range(len(grids)):
        temp = grids[i]
        pt1 = temp.grid_old.point1
        pt2 = temp.grid_old.point2
        pt3 = temp.grid_old.point3
        pt4 = temp.grid_old.point4
        if cal_position(pt1, data) or cal_position(pt2, data) or cal_position(pt3, data) or cal_position(pt4, data):
            gg.append(grids[i])
    points = []
    for i in range(len(gg)):
        points.append(gg[i].grid_new.point1)
        points.append(gg[i].grid_new.point2)
        points.append(gg[i].grid_new.point3)
        points.append(gg[i].grid_new.point4)
    return points


def sift_grids_inner(grids, data):
    '''
    get the grid tha in the original image
    :param grids:
    :return:
    '''
    # gg = []
    # for i in range (len(grids)):
    #     temp = grids[i]
    #     pt1 = temp.grid_old.point1
    #     pt2 = temp.grid_old.point2
    #     pt3 = temp.grid_old.point3
    #     pt4 = temp.grid_old.point4
    #     if(cal_position(pt1, data) or cal_position(pt2, data) or cal_position(pt3, data) or cal_position(pt4, data)):
    #         gg.append(grids[i])
    points = []
    for i in range(len(grids)):
        points.append(grids[i].grid_new.point1)
        points.append(grids[i].grid_new.point2)
        points.append(grids[i].grid_new.point3)
        points.append(grids[i].grid_new.point4)
    return points


def sift_grids2(grids):
    '''
    get the grid tha in the original image for GF and ZY3
    :param grids:
    :return:
    '''
    gg = []
    for i in range(len(grids)):
        temp = grids[i]
        pt1 = temp.grid_old.point1
        pt2 = temp.grid_old.point2
        pt3 = temp.grid_old.point3
        pt4 = temp.grid_old.point4
        if (cal_position(pt1, data) or cal_position(pt2, data) or cal_position(pt3, data) or cal_position(pt4, data)):
            gg.append(grids[i])
    points = []
    for i in range(len(gg)):
        points.append(gg[i].grid_new.point1)
        points.append(gg[i].grid_new.point2)
        points.append(gg[i].grid_new.point3)
        points.append(gg[i].grid_new.point4)
    return points


def cal_chunk(grids, data):
    chunk = []
    for m in tqdm.tqdm(range(len(grids))):
        # for m in range(len(grids)):
        y_dis = abs(grids[m].grid_new.point3.y - grids[m].grid_new.point1.y)
        x_dis = abs(grids[m].grid_new.point2.x - grids[m].grid_new.point1.x)

        startx = grids[m].grid_new.point3.x
        starty = grids[m].grid_new.point3.y

        a1, a2, a3, b1, b2, b3 = grids[m].cof[0:6]
        a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)

        new = np.zeros((int(y_dis) + 1, int(x_dis) + 1), dtype=np.int16)
        for i in range(int(y_dis) + 1):
            for j in range(int(x_dis) + 1):
                # print("m, i and j:", m, i, j)
                temppt = Grid_aff_interpolation(a1, a2, a3, b1, b2, b3, starty + i, startx + j)
                # print("point:",temppt.x, temppt.y)
                new[i][j] = bilinear_interpolation(temppt, data)
        b = np.array(list(new))
        chunk.append(b)
        # print("Num ", m, "chunk", 'percent: {:.2%}'.format(m / len(grids)))
    return chunk


def cal_chunk_R(grids, data, rpc1, rpc2):
    '''
    逐像素物方采样
    :param grids:
    :param data:
    :param rpc1:
    :param rpc2:
    :return:
    '''
    chunk = []
    for m in tqdm.tqdm(range(len(grids))):
        # for m in range(len(grids)):
        y_dis = abs(grids[m].grid_new.point3.y - grids[m].grid_new.point1.y)
        x_dis = abs(grids[m].grid_new.point2.x - grids[m].grid_new.point1.x)

        startx = grids[m].grid_new.point3.x
        starty = grids[m].grid_new.point3.y

        a1, a2, a3, b1, b2, b3 = grids[m].cof[0:6]
        a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)

        new = np.zeros((int(y_dis) + 1, int(x_dis) + 1), dtype=np.int16)
        for i in range(int(y_dis) + 1):
            print(i)
            for j in range(int(x_dis) + 1):
                # print("m, i and j:", m, i, j)
                temppt = Grid_aff_interpolation(a1, a2, a3, b1, b2, b3, starty + i, startx + j)
                # Rp = get_Rpoint(rpc1, rpc2, temppt, 0)
                # print("point:",temppt.x, temppt.y)
                new[i][j] = bilinear_interpolation(temppt, data)
        b = np.array(list(new))
        chunk.append(b)
        # print("Num ", m, "chunk", 'percent: {:.2%}'.format(m / len(grids)))
    return chunk


def get_Rpoint(rpc1, rpc2, initial_point, h):
    """
    """
    temp_point = point()
    lon, lat = rpc.RPC_Negative(initial_point.y, initial_point.x, h, rpc1)
    temp_point.y, temp_point.x = rpc.RPC_Positive(lon, lat, h, rpc2)
    return temp_point


def Poly2_resample(data, y_dis, x_dis, cof):
    new = np.zeros((int(y_dis) + 1, int(x_dis) + 1), dtype=np.int16)
    a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6 = cof[0:12]
    a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)
    a4, a5, a6, b4, b5, b6 = float(a4), float(a5), float(a6), float(b4), float(b5), float(b6)
    for i in range(y_dis):
        for j in range(x_dis):
            temppt = Poly2_interpolation(a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, i, j)
            new[i][j] = bilinear_interpolation(temppt, data)
    return new


def Poly3_resample(data, y_dis, x_dis, cof):
    new = np.zeros((int(y_dis) + 1, int(x_dis) + 1), dtype=np.int16)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 = cof[0:20]
    a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)
    a4, a5, a6, b4, b5, b6 = float(a4), float(a5), float(a6), float(b4), float(b5), float(b6)
    a7, a8, a9, a10 = float(a7), float(a8), float(a9), float(a10)
    b7, b8, b9, b10 = float(b7), float(b8), float(b9), float(b10)
    for i in range(y_dis):
        for j in range(x_dis):
            temppt = Poly3_interpolation(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9,
                                         b10, i, j)
            new[i][j] = bilinear_interpolation(temppt, data)
    return new


if __name__ == "__main__":
    exit()
