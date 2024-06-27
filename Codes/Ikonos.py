import numpy as np
import os
import rpc
from osgeo import gdal
import relocate
import math
import matplotlib .pyplot as plt
import time
import Polytran_and_resample
from filesave import Save_as_tif
from Polytran_and_resample import cal_chunk
import multiprocessing
from Parallel import cal_chunks
from Coordinate import Cal_epipolar_geotrans_L
from Coordinate import Cal_epipolar_geotrans_R
from epipolar_points_generation import generate_epipolar_points
from fileopen import readimg


if __name__ == "__main__":
    left = ""
    right = ""
    t1 = time.time()
    datasetL, datasetR, dataL, dataR, rpc1, rpc2 = readimg(Left, Right)
    t2 = time.time()
    print("Read Img time:", t2 - t1, "s")

    t1 = time.time()
    LL_points, LL_points_all,  RR_points, RR_points_all, initial, R_start_point, L_coe, R_coe = generate_epipolar_points(dataL, dataR, rpc1, rpc2, h_Max=2500, h_Min=0)
    t2 = time.time()
    print("generate_epipolar_points:", t2 - t1, "s")

    t1 = time.time()
    mmm, nnn = relocate.relocate_sub(initial, LL_points_all, L_coe[0][0], R_start_point, RR_points_all, R_coe[0][0])
    t2 = time.time()
    # print("Rolate is OK!")
    print("epipolar points rolation time:", t2 - t1, "s")

    t1 = time.time()
    L_grids = Polytran_and_resample.Normalize_grids(LL_points_all, mmm)
    R_grids = Polytran_and_resample.Normalize_grids(RR_points_all, nnn)
    # print("Grids calculated OK!")
    # mmm_in = Polytran_and_resample.sift_grids(L_grids, dataL)
    # nnn_in = Polytran_and_resample.sift_grids(R_grids, dataR)
    #
    # poly_L, poly_R, x_cof, y_range = Polytran_and_resample.Determin_region(mmm_in, nnn_in)
    # L_grids = L_grids[0:10]

    # L_tans = Cal_epipolar_geotrans_L(datasetL, L_grids, x_cof, y_range)
    # R_tans = Cal_epipolar_geotrans_R(datasetR, R_grids, x_cof, y_range)
    t2 = time.time()
    print("Calc Aff_coff time:", t2 - t1, "s")

    # L_img = "K:\DSM\ikonos_epipolor\L_img_stero.tiff"
    # datasetL = gdal.Open(L_img, gdal.GA_ReadOnly)
    # data = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
    # Save_as_tif(Left, "K:\DSM\ikonos_epipolor\geo_L.tiff", data, band=1,trans=L_tans)
    # print("Save OK!")
    #
    # R_img = "K:\DSM\ikonos_epipolor\R_img_stero.tiff"
    # datasetR = gdal.Open(R_img, gdal.GA_ReadOnly)
    # data = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)
    # Save_as_tif(Right, "K:\DSM\ikonos_epipolor\geo_R.tiff", data, band=1, trans=R_tans)
    # print("Save OK!")

    print("L_grids_num", len(L_grids))
    mmm_in = Polytran_and_resample.sift_grids(L_grids, dataL)
    nnn_in = Polytran_and_resample.sift_grids(R_grids, dataR)
    xx = L_grids[0].grid_new.point4.x - L_grids[0].grid_new.point1.x
    yy = L_grids[0].grid_new.point4.y - L_grids[0].grid_new.point1.y
    print("L_grids X", xx)
    print("L_grids Y", yy)

    newLL = np.zeros((30000, 30000), dtype=np.int16)
    # chunk_L = cal_chunk(L_grids, dataL)
    startt = time.time()
    chunk_L = cal_chunk(L_grids, dataL)
    endt = time.time()
    print("L_img Interpolation time:", endt - startt, "s")
    # t1 = time.time()
    # newLL = Polytran_and_resample.value_chunk(chunk_L, newLL, L_grids)
    # newLL = Polytran_and_resample.cut_minpolygon_L(newLL, x_cof, y_range)
    # Save_as_tif(Left, "K:\IKONOS\L_img_stero.tiff", newLL, band=1)
    # t2 = time.time()
    # print("left epipolar image is saved! time:", t2 - t1, "s")
    #
    # newRR = np.zeros((30000, 30000), dtype=np.int16)
    # t1 = time.time()
    # chunk_R = cal_chunks(R_grids, dataR)
    # t2 = time.time()
    # print("R_img Interpolation time:", t2 - t1, "s")
    # t1 = time.time()
    # newRR = Polytran_and_resample.value_chunk(chunk_R, newRR, R_grids)
    # newRR = Polytran_and_resample.cut_minpolygon_R(newRR, x_cof, y_range)
    # Save_as_tif(Right, "K:\IKONOS\R_img_stero.tiff", newRR, band=1)
    # t2 = time.time()
    # print("right epipolar image is saved! time:", t2 - t1, "s")