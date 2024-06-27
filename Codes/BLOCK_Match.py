import os
import numpy as np
from osgeo import gdal
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from matplotlib import pyplot as plt
from SIFT_BF import chunk_Sift
gdal.DontUseExceptions()

def chunk_math(img1, img2, min, range):
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    blockSize = 5
    img_channels = 3
    min = int(min)
    range = int(range)
    stereo = cv2.StereoSGBM_create(minDisparity=min,
                                   numDisparities=range,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=1,
                                   preFilterCap=15,
                                   uniquenessRatio=0,
                                   speckleWindowSize=2,
                                   speckleRange=63,
                                   mode=cv2.STEREO_SGBM_MODE_HH4)
    # 计算视差
    disparity = stereo.compute(img1, img2)
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

if __name__ == "__main__":
    print("BLOCK_Match!")

    Left = "K:\DSM\ikonos_epipolor\R_img_stero.tiff"
    Right = “K:\DSM\ikonos_epipolor\R_img_stero.tiff"

    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)

    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)

    dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize )
    # tempL = datasetL.GetMetadata("RPC")
    # tempR = datasetR.GetMetadata("RPC")
    # rpc1 = rpc.pre_process_RPCL(tempL)
    # rpc2 = rpc.pre_process_RPCR(tempR)
    print("Successfully read Images")

    # img1 = dataL[0:500,700:1700]
    # img2 = dataR[0:500,1600:2600]
    # plt.imshow(img1)
    # plt.close()
    # plt.imshow(img2)
    # plt.close()
    # img1 = cv2.resize(dataL, fx = 0.05, fy = 0.05, dsize = None)
    # img2 = cv2.resize(dataR, fx =0.05, fy =0.05, dsize = None)
    rownum = 256
    colnum = 2048 + 1024
    rowt = int(dataL.shape[0]/rownum)
    colt = int((dataL.shape[1]-1000)/(1024))-2
    coliniL = 1000
    coliniR = 0
    disparities = np.zeros((dataL.shape[0], dataL.shape[1]), dtype=np.int16)
    for i in range(0,rowt):
        for j in range(0, colt):

            img1 = dataL[  rownum*i:rownum*(i+1), coliniL+(colnum-2048)*j:(colnum-2048)*(j)+coliniL+colnum]
            img2 = dataR[  rownum*i:rownum*(i+1), coliniR+(colnum-2048)*j:(colnum-2048)*(j)+coliniR+colnum]
            print("rowt", i, "colt", j, "IMG1",img1.shape,"IMG2",img2.shape)
            # chunk_Sift(img1, img2)
            # cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
            # cv2.imshow("img1", cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"))
            #
            # cv2.imshow("img2",cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            disparity = chunk_math(img1, img2, -1024, 2048)
            # plt.imshow(disparity)
            disparities[rownum*i:rownum*(i+1), coliniL+(colnum-2048)*j+1024:(colnum-2048)*(j)+coliniL+2048 ] = disparity[0:256,1024:2048]
    # cv2.imwrite("Disparity.png", disparities)

    driver = gdal.GetDriverByName("GTiff")
    etype = gdal.GDT_Int16
    ds = driver.Create("K:\GF7\epipolar\disparity4.tiff", disparities.shape[1], disparities.shape[0], 1, etype)
    ds.GetRasterBand(1).WriteArray(disparities)
    ds.FlushCache()
    del ds


    # # # 归一化函数算法，生成深度图（灰度图）
    # disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # plt.imshow(disp, 'gray')
    # plt.show()
    #
    # # 生成深度图（颜色图）
    # dis_color = disparity
    # dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # dis_color = cv2.applyColorMap(dis_color, 2)
    #
    # cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    # cv2.imshow("depth", dis_color)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


