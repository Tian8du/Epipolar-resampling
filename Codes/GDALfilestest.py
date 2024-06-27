#  导入所需要的包
from osgeo import gdal
import numpy as np
import cv2
import matplotlib .pyplot as plt

#  读取遥感影像，获取影像的基本信息
def read_tif(filename, b=1):

    data = gdal.Open(filename)
    driver = gdal.GetDriverByName('GTiff')
    # driver = data.GetDriver()

    # 数据集的基本信息
    print('Raster Driver : {d}\n'.format(d=driver.ShortName))
    img_width, img_height = data.RasterXSize, data.RasterYSize
    print('影像的列，行数: {r}rows * {c}colums'.format(r=img_width, c=img_height))
    print('栅格数据的空间参考：{}'.format(data.GetGeoTransform()))  # 栅格数据的6参数
    print('投影信息：{}\n'.format(data.GetProjection()))  # 栅格数据的投影

    # 读取影像的元数据,获取各个波段的信息
    print(data.GetMetadata())
    band = None
    for i in range(b):
        band_1 = data.GetRasterBand(i + 1)
        band_1 = band_1.ReadAsArray(0, 0, img_width, img_height).flatten()  # 行，列
        if band is None:
            band = band_1
        else:
            band = np.vstack((band, band_1))
        return band.T


def arr2raster(orginal_file, result_file, arr, band=1):
     '''
         orginal_file为原始影像；目的为获取原始影像的地理参考。
         result_file为要保存的文件的名称，arr为转存的矩阵，proj和transform分别为投影信息和仿射变换矩阵
         band=1,表示默认情况生成单波段图像，若要多波段图像，在输入的时候更改band为要创建的波段数
     '''
     driver = gdal.GetDriverByName("GTiff")
     etype = gdal.GDT_Int16
     proj = gdal.Open(orginal_file).GetProjection()
     transform = gdal.Open(orginal_file).GetGeoTransform()

     # no_data_value = ''
     if band == 1:
         ds = driver.Create(result_file, arr.shape[1], arr.shape[0], band, etype)        # 行，列
         # 写单波段图像
         ds.GetRasterBand(1).WriteArray(arr)
     else:
         ds = driver.Create(result_file, arr.shape[2], arr.shape[1], band, etype)
         # 写多波段图像
         for i in range(band):
             ds.GetRasterBand(i + 1).WriteArray(arr[i, :, :])
     ds.SetGeoTransform(transform)
     ds.SetProjection(proj)
     ds.FlushCache()
     del ds

def Read_img2array(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    print('处理图像波段数总共有：', dataset.RasterCount)
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    geotrans = dataset.GetGeoTransform()  # 几何信息
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
   # 直接读取dataset
    img_array = dataset.ReadAsArray()
    return im_width,im_height,im_bands,projection, geotrans, img_array



# tiff_file为tif,im_data为数组
def Write_img2array(tiff_file, im_proj, im_geotrans, data_array):
    if 'int8' in data_array.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'int16' in data_array.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32

    if len(data_array.shape) == 3:
        im_bands, im_height, im_width = data_array.shape
    else:
        im_bands, (im_height, im_width) = 1, data_array.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(tiff_file, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data_array)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data_array[i])
    del dataset





