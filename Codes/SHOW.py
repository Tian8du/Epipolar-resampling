# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal
import cv2


class IMAGE:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        del dataset

        return im_bands, im_data

    # 写GeoTiff文件
    def write_img(self, filename, im_proj, im_geotrans, im_data):

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


## 影像拉伸 ##
def img_processing(im_band, img_data):
    if im_band == 1:
        data_jpg = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')
        im_max = np.amax(img_data)
        im_min = np.amin(img_data)
        for m in range(0, img_data.shape[0]):
            for n in range(0, img_data.shape[1]):
                data_jpg[m, n] = float(255. / (im_max - im_min)) * (img_data[m, n] - im_min)
    else:
        data_jpg = np.zeros((img_data.shape[1], img_data.shape[2], 3), dtype='uint8')
        for i in range(3):
            im_max = np.amax(img_data[i, :, :])
            im_min = np.amin(img_data[i, :, :])
            for m in range(0, img_data.shape[1]):
                for n in range(0, img_data.shape[2]):
                    data_jpg[m, n, i] = float(255. / (im_max - im_min)) * (img_data[i, m, n] - im_min)
    return data_jpg


## GAMMA 变换 ##
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)




if __name__ == '__main__':
    image_grid = IMAGE()
    image_name = 'K:/data/GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff'
    im_band, img_data = image_grid.read_img(image_name)
    data_jpg = img_processing(im_band, img_data)
    data_jpg_corrted = gamma_trans(data_jpg, 0.5)
    ## 显示 ##
    cv2.imshow('K:/data/GF7_DLC_E102.8_N27.3_20210330_L1A0000379699-BWDPAN.tiff', data_jpg)
    cv2.imshow('Gamma_image', data_jpg_corrted)
    cv2.waitKey()