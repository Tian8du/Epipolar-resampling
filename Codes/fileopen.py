import rpc
from osgeo import gdal


def readimg(Left, Right):
    '''
    Read satellite imgs from stereo imgs.
    :param Left: Left img
    :param Right: Right img
    :return: GDAL dataset.
    '''
    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)
    dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)
    tempL = datasetL.GetMetadata("RPC")
    tempR = datasetR.GetMetadata("RPC")
    rpc1 = rpc.pre_process_RPCL(tempL)
    rpc2 = rpc.pre_process_RPCR(tempR)
    return datasetL, datasetR, dataL, dataR, rpc1, rpc2


def readimg2(Left, Right):
    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    dataL = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)
    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)
    dataR = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)
    return dataL, dataR
