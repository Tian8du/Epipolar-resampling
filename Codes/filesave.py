from osgeo import gdal
import numpy as np
import cv2
import matplotlib .pyplot as plt


def Save_as_tif(orginal_file, result_file, arr, band=1, trans=None):
    '''
        orginal_file为原始影像；目的为获取原始影像的地理参考。
        result_file为要保存的文件的名称，arr为转存的矩阵，proj和transform分别为投影信息和仿射变换矩阵
        band=1,表示默认情况生成单波段图像，若要多波段图像，在输入的时候更改band为要创建的波段数
    '''
    driver = gdal.GetDriverByName("GTiff")
    etype = gdal.GDT_UInt16
    proj = gdal.Open(orginal_file).GetProjection()


    # no_data_value = ''
    if band == 1:
        ds = driver.Create(result_file, arr.shape[1], arr.shape[0], band, etype)  # 行，列
        # 写单波段图像
        ds.GetRasterBand(1).WriteArray(arr)
    else:
        ds = driver.Create(result_file, arr.shape[1], arr.shape[0], band, etype)
        # 写多波段图像
        ds.GetRasterBand(band).WriteArray(arr)
    # ds.SetGeoTransform(trans)
    ds.SetProjection(proj)
    ds.FlushCache()
    del ds

if __name__ == "__main__":
    L_img = "K:\DSM\ikonos_epipolor\L_img_stero.tiff"
    datasetL = gdal.Open(L_img, gdal.GA_ReadOnly)
    data = datasetL.ReadAsArray(7000, 7000, 1024, 1024)
    Save_as_tif(L_img, "K:\DSM\ikonos_epipolor\partL1024.tiff", data, band=1)
    print("Save OK!")

    R_img = "K:\DSM\ikonos_epipolor\R_img_stero.tiff"
    datasetR = gdal.Open(R_img, gdal.GA_ReadOnly)
    data = datasetR.ReadAsArray(7000, 7000, 1024, 1024)
    Save_as_tif(R_img, "K:\DSM\ikonos_epipolor\partR1024.tiff", data, band=1)
    print("Save OK!")