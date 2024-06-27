from SIFT_BF import chunk_Sift
import cv2


if __name__ == "__main__":

    left_img = cv2.imread("K:\DSM\ikonos_epipolor\geo_L.tiff", 2)
    right_img = cv2.imread("K:\DSM\ikonos_epipolor\geo_R.tiff", 2)
    for i in range(10):
        for j in range(10):
            img1 = left_img[i*1000:(i+1)*1000,j*1000:(j+1)*1000]
            img2 = right_img[i*1000:(i+1)*1000,j*1000:(j+1)*1000]
            chunk_Sift(img1, img2)