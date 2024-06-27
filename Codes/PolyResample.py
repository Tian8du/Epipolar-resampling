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
import Basefunction

if __name__ == "__main__":
    # Left = "K:\DSM\po_162796_0000000\po_162796_0000000\po_162796_pan_0000000.tif"
    # Right = "K:\DSM\po_162796_0010000\po_162796_0010000\po_162796_pan_0010000.tif"
    # Left = "K:\DSM\ZY3_test\ZY3_01a_mynbavp_278116_20140827_183905_0008_SASMAC_CHN_sec_rel_001_14082905653.tif"
    # Right = "K:\DSM\ZY3_test\ZY3_01a_mynfavp_278116_20140827_183807_0008_SASMAC_CHN_sec_rel_001_14082905709.tif"
    Left = "K:\GF7\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-FWDPAN.tiff"
    Right = "K:\GF7\GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff"
    t1 = time.time()
    datasetL, datasetR, dataL, dataR, rpc1, rpc2 = readimg(Left, Right)
    t2 = time.time()
    print("Read Img time:", t2 - t1, "s")

    t1 = time.time()
    LL_points, LL_points_all, RR_points, RR_points_all, initial, R_start_point, L_coe, R_coe = generate_epipolar_points(
        dataL, dataR, rpc1, rpc2, h_Max=2500, h_Min=0)
    t2 = time.time()
    print("generate_epipolar_points:", t2 - t1, "s")

    t1 = time.time()
    mmm, nnn = relocate.relocate_sub2(initial, LL_points, L_coe[0][0], R_start_point, RR_points, R_coe[0][0])
    t2 = time.time()
    # print("Rolate is OK!")
    print("epipolar points rolation time:", t2 - t1, "s")

    LL_points = Basefunction.extra_points_from_list(LL_points)
    RR_points = Basefunction.extra_points_from_list(RR_points)
    mmm = Basefunction.extra_points_from_list(mmm)
    nnn = Basefunction.extra_points_from_list(nnn)
    t1 = time.time()
    cof_L = Basefunction.cal_coff_aff3(LL_points, mmm)
    cof_R = Basefunction.cal_coff_aff3(RR_points, nnn)
    t2 = time.time()
    print("Cal poly_cof time:", t2 - t1, "s")
    poly_L, poly_R, x_cof, y_range = Polytran_and_resample.Determin_region(mmm, nnn)
    print("OK")
    y_dis = int(poly_L[3]) + 1
    x_dis = int(poly_L[1]) + 1
    t1 = time.time()
    L_epipolar_img = Polytran_and_resample.Poly3_resample(dataL, y_dis, x_dis, cof_L)
    t2 = time.time()
    print("Interpolate pixel by Poly3", t2 - t1, "s")

    # t1 = time.time()
    # Save_as_tif(Left, "K:\Paper_data\ZY3L_img_stero.tiff", L_epipolar_img, band=1)
    # t2 = time.time()
    print()
