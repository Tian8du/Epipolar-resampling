import os
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from matplotlib import pyplot as plt
from fileopen import readimg2
from osgeo import gdal

def chunk_Sift(img1, img2):
    '''
    Cal the y_parall of a chunk
    :param img1: A image of left
    :param img2: A image of right
    :return: some pars of y_parall using SIFT
    '''
    # Normalize image to uint8
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    # img1 = img1.T
    # img2 = img2.T
    # 创建SIFT算子
    sift = cv2.SIFT_create()
    # 检测特征点与描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 创建蛮力（BF）匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)


    leftimges = []
    rigtimages = []
    # print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.5 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            # print(i, pt1, pt2, pt1[1] - pt2[1])
            leftimges.append(pt1)
            rigtimages.append(pt2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    sumy = 0
    pty = []
    count = 0
    ptx = []
    for j in range(len(leftimges)):
        cut = abs(leftimges[j][1] - rigtimages[j][1])
        pty.append(cut)
        cutx = leftimges[j][0] - rigtimages[j][0]
        ptx.append(cutx)

        # if (cut > 1):
        #     print("Over", leftimges[j], rigtimages[j])
        #     count = count + 1

        sumy = sumy + (leftimges[j][1] - rigtimages[j][1])
    if len(leftimges) == 0:
        print("No math point")
        return 0

    ave_y = sumy / len(leftimges)
    print("ave_y", ave_y,"Max_ycut", max(pty),"Min_ycut", min(pty))
    # print( "Max_xcut", max(ptx), "Min_ycut", min(ptx))
    #show sift

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    # cv2.imshow("show", img3)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print("Max_ycut", max(pty))
    # print("Min_ycut", min(pty))
    # print("all_count", len(leftimges))
    # print("ycut>1:num", count, 'percent: {:.2%}'.format(count / len(leftimges)))

    # print("cutX_max", max(ptx), "MIN", min(ptx))





if __name__ == "__main__":
    # img1 = cv2.imread("K:\DSM\ikonos_epipolor\L_img_stero.tiff", 2)
    # img2 = cv2.imread("K:\DSM\ikonos_epipolor\R_img_stero.tiff", 2)
    # img1, img2 = readimg2("K:\data\GF\L_img.tiff", "K:\data\GF\R_img.tiff")
    # img1 = cv2.imread("K:\\baihetanZY3\ZY302_TMS_E102.8_N27.1_20221126_L1A0001074472\DSM\L_epipolar_img.tiff",2)
    # img2 = cv2.imread("K:\\baihetanZY3\ZY302_TMS_E102.8_N27.1_20221126_L1A0001074472\DSM\R_epipolar_img.tiff",2)
    # img1 = cv2.imread("K:\DSM\ZY3_test\L_epipolar_img.tiff", 2)
    # img2 = cv2.imread("K:\DSM\ZY3_test\R_epipolar_img.tiff", 2)
    # img1 = cv2.imread("K:\DSM\ZY3_new_test\L_img_stero.tiff",2)
    # img2 = cv2.imread("K:\DSM\ZY3_new_test\R_img_stero.tiff",2)
    # # img1 = cv2.resize(img1, fx = 0.05, fy = 0.05, dsize = None)
    # # img2 = cv2.resize(img2, fx =0.05, fy =0.05, dsize = None)
    # img1 = cv2.imread("K:\ZY3_SPRS\Epipolar\L_epipolar_img4.tiff",2)
    # img2 = cv2.imread("K:\ZY3_SPRS\Epipolar\R_epipolar_img4.tiff",2)
    # img1 = cv2.imread("E:\RPCDSM\\venv\RPCline\left.png",2)
    # img2 = cv2.imread("E:\RPCDSM\\venv\RPCline\Right.png",2)
    # img1 = cv2.imread("K:\WHDSM\L_epipolar_img.tiff",2)
    # img2 = cv2.imread("K:\WHDSM\R_epipolar_img.tiff",2)
    Left = "K:\GF7\epipolar\L_epipolar.tiff"
    Right = "K:\GF7\epipolar\R_epipolar.tiff"
    datasetL = gdal.Open(Left, gdal.GA_ReadOnly)
    img1 = datasetL.ReadAsArray(0, 0, datasetL.RasterXSize, datasetL.RasterYSize)

    datasetR = gdal.Open(Right, gdal.GA_ReadOnly)

    img2 = datasetR.ReadAsArray(0, 0, datasetR.RasterXSize, datasetR.RasterYSize)

    # img1 = cv2.resize(img1, fx = 0.1, fy = 0.1, dsize = None)
    # img2 = cv2.resize(img2, fx =0.1, fy =0.1, dsize = None)
    img1 = img1[1500:3500, 22000:24000]
    img2 = img2[1500:3500, 21300:23300]

    # img1 = img1[600:800,600:800]
    # img2 = img2[600:800,602:802]
    # temp = img1
    # img1 = img2
    # img2 = temp
    # img1 = img1.T
    # img2 = img2.T
    img1=cv2.normalize(img1,None,0,255,cv2.NORM_MINMAX).astype("uint8")
    img2=cv2.normalize(img2,None,0,255,cv2.NORM_MINMAX).astype("uint8")

    # cv2.imwrite("K:\DSM\ikonos_epipolor\Stero\img1solu.jpg", img1)
    # cv2.imwrite("K:\DSM\ikonos_epipolor\Stero\img2solu.jpg", img2)
    sift = cv2.SIFT_create()

    # 检测特征点与描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建蛮力（BF）匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 比值测试，首先获取与 A 距离最近的点 B（最近）和 C（次近），只有当 B/C
    # 小于阈值时（ 0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为 0
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])

    leftimges = []
    rigtimages = []
    print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.4 * m2.distance:# 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            print(i, pt1, pt2,pt1[1]-pt2[1])
            leftimges.append(pt1)
            rigtimages.append(pt2)

    sumy = 0
    pty = []
    count = 0
    ptx = []
    for j in range(len(leftimges)):
        cut = leftimges[j][1] - rigtimages[j][1]
        # cut = abs(leftimges[j][1] - rigtimages[j][1])
        pty.append(cut)
        cutx = leftimges[j][0] - rigtimages[j][0]
        ptx.append(cutx)

        if( abs(cut )> 1):
            print("Over",leftimges[j],rigtimages[j])
            count = count + 1

        sumy = sumy + abs(leftimges[j][1] - rigtimages[j][1])

    ave_y = sumy/len(leftimges)
    print("ave_y",ave_y)
    print("Max_ycut",max(pty))
    print("Min_ycut",min(pty))
    print("all_count", len(leftimges))
    print("ycut>1:num", count, 'percent: {:.2%}'.format(count / len(leftimges)))

    print("cutX_max",max(ptx),"MIN",min(ptx))


    # cv.drawMatchesKnn()把列表作为匹配项。
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.namedWindow('show',cv2.WINDOW_NORMAL)
    cv2.imshow("show", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()

    blockSize = 5
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=-256,
                                    numDisparities= 512,
                                    blockSize=blockSize,
                                    P1=8 * img_channels * blockSize * blockSize,
                                    P2=32* img_channels * blockSize * blockSize,
                                    disp12MaxDiff=1,
                                    preFilterCap=15,
                                    uniquenessRatio=0,
                                    speckleWindowSize=2,
                                    speckleRange=63,
                                    mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1, img2)
    disparity = disparity.astype(np.float32) /16
    plt.imshow(disparity,'gray')


    # # # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.imshow(disp,'gray')
    plt.show()

    driver = gdal.GetDriverByName("GTiff")
    etype = gdal.GDT_Int16
    ds = driver.Create("K:\GF7\epipolar\GF7disparity.tiff", disparity.shape[1], disparity.shape[0], 1, etype)
    ds.GetRasterBand(1).WriteArray(disparity)
    ds.FlushCache()
    del ds



    # 生成深度图（颜色图）
    dis_color = disp
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    cv2.namedWindow('depth',cv2.WINDOW_NORMAL)
    cv2.imshow("depth", dis_color)
    cv2.imwrite("K:\DSM\ikonos_epipolor\Stero\disparity3.jpg", disparity)

    cv2.waitKey()
    cv2.destroyAllWindows()